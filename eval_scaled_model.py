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

def main():
    args = parse_arguments()
    args.save = args.save or f"/kaggle/working/checkpoints_{args.exp_name or 'default'}"
    args.checkpoints_path = args.save
    args.results_dir = args.save.replace("checkpoints", "results")
    args.data_location = "/kaggle/working/datasets"
    os.makedirs(args.results_dir, exist_ok=True)

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    # 🔹 Load previously selected best alpha
    with open(os.path.join(args.results_dir, "task_addition_results.json")) as f:
        task_addition_results = json.load(f)
    best_alpha = task_addition_results["alpha"]

    task_vectors = [NonLinearTaskVector(
        pretrained_checkpoint=os.path.join(args.checkpoints_path, "pretrained.pt"),
        finetuned_checkpoint=os.path.join(args.checkpoints_path, f"{ds}_finetuned.pt")
    ) for ds in datasets]

    scaled_results = {
        "alpha": best_alpha,
        "Single-task Acc. (Train)": [],
        "Single-task Acc. (Test)": [],
        "logTr[F̂·] (Train)": []
    }

    print(f"🔁 Using best alpha = {best_alpha:.2f}")
    print("🔬 Evaluating each task with scaled τₜ:")

    for i, dataset in enumerate(datasets):
        print(f"📌 {dataset}...")

        encoder = (task_vectors[i] * best_alpha).apply_to(
            os.path.join(args.checkpoints_path, "pretrained.pt"), scaling_coef=1.0
        ).cuda()

        path = resolve_dataset_path(args, dataset)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3) if dataset.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            train_ds = get_dataset(f"{dataset}Val", preprocess, path, args.batch_size)
            train_loader = train_ds.train_loader
            head = get_classification_head(args, dataset).cuda()
            model = ImageClassifier(encoder, head).cuda()
            train_acc = evaluate_model(model, train_loader)
            fim_trace = train_diag_fim_logtr(args, model, dataset)

            test_ds = get_dataset(dataset, preprocess, path, args.batch_size)
            test_loader = test_ds.test_loader
            test_acc = evaluate_model(model, test_loader)

            scaled_results["Single-task Acc. (Train)"].append(train_acc)
            scaled_results["Single-task Acc. (Test)"].append(test_acc)
            scaled_results["logTr[F̂·] (Train)"].append(fim_trace)

        except Exception as e:
            print(f"⚠️ Error evaluating {dataset}: {e}")
            for key in scaled_results:
                if key != "alpha":
                    scaled_results[key].append(0.0)

    scaled_path = os.path.join(args.results_dir, "scaled_model_results.json")
    with open(scaled_path, "w") as f:
        json.dump(scaled_results, f, indent=4)
    print(f"✅ Scaled model results saved to: {scaled_path}")

if __name__ == "__main__":
    main()
