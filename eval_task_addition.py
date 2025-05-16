import os
import json
import torch
import numpy as np
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from task_vectors import NonLinearTaskVector
from utils import train_diag_fim_logtr
from torchvision import transforms

def resolve_dataset_path(args, dataset_name):
    base_path = args.data_location
    dataset_paths = {
        "dtd": base_path,
        "eurosat": base_path,
        "mnist": os.path.join(base_path, "MNIST", "raw"),
        "gtsrb": os.path.join(base_path, "gtsrb"),
        "resisc45": base_path,
        "svhn": os.path.join(base_path, "svhn"),
    }
    return dataset_paths.get(dataset_name.lower(), base_path)

def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def compute_average_normalized_accuracy(val_accuracies, single_task_accuracies):
    return np.mean([v / s if s != 0 else 0 for v, s in zip(val_accuracies, single_task_accuracies)])

def main():

    args = parse_arguments()
    # 🌍 Portable path setup
    project_root = os.path.abspath(args.data_location)
    args.data_location = os.path.join(project_root, "datasets")
    
    if args.save is None:
        if args.exp_name:
            args.save = os.path.join(project_root, f"checkpoints_{args.exp_name}")
        else:
            args.save = os.path.join(project_root, "checkpoints_default")
    
    args.checkpoints_path = args.save
    args.results_dir = args.save.replace("checkpoints", "results")
    os.makedirs(args.results_dir, exist_ok=True)

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    task_vectors = [NonLinearTaskVector(
        pretrained_checkpoint=os.path.join(args.checkpoints_path, "pretrained.pt"),
        finetuned_checkpoint=os.path.join(args.checkpoints_path, f"{ds}_finetuned.pt")
    ) for ds in datasets]

    single_task_accuracies = []
    train_accuracies = []
    for dataset in datasets:
        with open(os.path.join(args.results_dir, f"{dataset}_results.json")) as f:
            res = json.load(f)
        single_task_accuracies.append(res["test_accuracy"])
        train_accuracies.append(res["train_accuracy"])

    # Resume support for alpha sweep
    progress_path = os.path.join(args.results_dir, "alpha_search_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            alpha_results = json.load(f)
    else:
        alpha_results = {}

    
    for alpha in np.arange(0.0, 1.05, 0.05):
        alpha_str = f"{alpha:.2f}"
        if alpha_str in alpha_results:
            print(f"⏩ Skipping alpha = {alpha_str}, already computed.")
            continue
    
        print(f"🔁 Calculating alpha = {alpha_str}")
        combined_vector = sum((tv * alpha for tv in task_vectors))
        blended_encoder = combined_vector.apply_to(
            os.path.join(args.checkpoints_path, "pretrained.pt"), scaling_coef=1.0
        ).cuda()
        val_accuracies = []
    
        for dataset in datasets:
            try:
                path = resolve_dataset_path(args, dataset)
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(3) if dataset.lower() == "mnist" else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                val_ds = get_dataset(f"{dataset}Val", preprocess, path, args.batch_size)
                val_loader = val_ds.train_loader
                head = get_classification_head(args, dataset).cuda()
                model = ImageClassifier(blended_encoder, head).cuda()
                acc = evaluate_model(model, val_loader)
                val_accuracies.append(acc)
            except Exception as e:
                print(f"⚠️ Skipping {dataset} during alpha search due to error: {e}")
                val_accuracies.append(0.0)
    
        avg_score = compute_average_normalized_accuracy(val_accuracies, single_task_accuracies)
    
        # Store progress
        alpha_results[alpha_str] = {
            "val_accuracies": val_accuracies,
            "avg_normalized_score": avg_score
        }
        with open(progress_path, "w") as f:
            json.dump(alpha_results, f, indent=4)

  
    best_alpha = max(alpha_results.items(), key=lambda x: x[1]["avg_normalized_score"])[0]
    best_alpha = float(best_alpha)
    print(f"🏆 Best alpha: {best_alpha:.2f}")


    # Final evaluation with best alpha
    combined_vector = sum((tv * best_alpha for tv in task_vectors))
    blended_encoder = combined_vector.apply_to(
        os.path.join(args.checkpoints_path, "pretrained.pt"), scaling_coef=1.0
    ).cuda()
    results = {"alpha": best_alpha}

    for mode in ["train", "test"]:
        absolute_acc, normalized_acc, fim_traces = [], [], []
        for i, dataset in enumerate(datasets):
            try:
                path = resolve_dataset_path(args, dataset)
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(3) if dataset.lower() == "mnist" else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                ds = get_dataset(dataset, preprocess, path, args.batch_size)
                loader = ds.train_loader if mode == "train" else ds.test_loader
                head = get_classification_head(args, dataset).cuda()
                model = ImageClassifier(blended_encoder, head).cuda()
                acc = evaluate_model(model, loader)
                absolute_acc.append(acc)
                base_acc = train_accuracies[i] if mode == "train" else single_task_accuracies[i]
                normalized_acc.append(acc / base_acc if base_acc != 0 else 0)
                fim = train_diag_fim_logtr(args, model, dataset)
                fim_traces.append(fim)
            except Exception as e:
                print(f"⚠️ Skipping {dataset} in {mode} mode due to error: {e}")
                absolute_acc.append(0.0)
                normalized_acc.append(0.0)
                fim_traces.append(0.0)

        results[f"absolute_{mode}_accuracy"] = absolute_acc
        results[f"normalized_{mode}_accuracy"] = normalized_acc
        if mode == "train":
            results["fim_log_traces_train"] = fim_traces

    with open(os.path.join(args.results_dir, "task_addition_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Task addition results saved to {args.results_dir}/task_addition_results.json")


    # ✅ Evaluate each individual scaled τₜ: f(θ₀ + α⋅τₜ)
    scaled_results = {
        "alpha": best_alpha,
        "Single-task Acc. (Train)": [],
        "Single-task Acc. (Test)": [],
        "logTr[F̂·] (Train)": []
    }

    print("\n🔬 Evaluating each task with its own scaled τₜ (after scaling)...")
    for i, dataset in enumerate(datasets):
        print(f"📌 Evaluating scaled τₜ for {dataset}...")

        # Apply alpha-scaled single task vector
        encoder = (task_vectors[i] * best_alpha).apply_to(
            os.path.join(args.checkpoints_path, "pretrained.pt"), scaling_coef=1.0
        ).cuda()

        # Common preprocessing
        path = resolve_dataset_path(args, dataset)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3) if dataset.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            # Train Evaluation
            train_ds = get_dataset(f"{dataset}Val", preprocess, path, args.batch_size)
            train_loader = train_ds.train_loader
            head = get_classification_head(args, dataset).cuda()
            model = ImageClassifier(encoder, head).cuda()

            train_acc = evaluate_model(model, train_loader)
            fim_trace = train_diag_fim_logtr(args, model, dataset)

            # Test Evaluation
            test_ds = get_dataset(dataset, preprocess, path, args.batch_size)
            test_loader = test_ds.test_loader
            test_acc = evaluate_model(model, test_loader)

            # Save required metrics only
            scaled_results["Single-task Acc. (Train)"].append(train_acc)
            scaled_results["Single-task Acc. (Test)"].append(test_acc)
            scaled_results["logTr[F̂·] (Train)"].append(fim_trace)

        except Exception as e:
            print(f"⚠️ Error evaluating {dataset} after scaling: {e}")
            for key in scaled_results:
                if key != "alpha":
                    scaled_results[key].append(0.0)

    # ✅ Save scaled results
    scaled_path = os.path.join(args.results_dir, "scaled_model_results.json")
    with open(scaled_path, "w") as f:
        json.dump(scaled_results, f, indent=4)
    print(f"✅ Scaled model results saved to {scaled_path}")


if __name__ == "__main__":
    main()
