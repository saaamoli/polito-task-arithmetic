import os
import json
import torch
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from args import parse_arguments
from utils import train_diag_fim_logtr as compute_fim_log_trace
from torchvision import transforms

def resolve_dataset_path(args, dataset_name):
    base_path = args.data_location
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "dtd":
        return os.path.join(base_path, "dtd")
    elif dataset_name_lower == "eurosat":
        return base_path
    elif dataset_name_lower == "mnist":
        return os.path.join(base_path, "MNIST", "raw")
    elif dataset_name_lower == "gtsrb":
        return os.path.join(base_path, "gtsrb")
    elif dataset_name_lower == "resisc45":
        return base_path
    elif dataset_name_lower == "svhn":
        return os.path.join(base_path, "svhn")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def evaluate_scaled_model():
    args = parse_arguments()
    args.model = "ViT-B-32__pretrained__openai"
    args.cache_dir = None
    args.openclip_cachedir = None
    args.checkpoints_path = "/kaggle/working/checkpoints_batchsize"
    args.results_dir = "/kaggle/working/results_after_scaling64"
    args.save = args.results_dir
    args.device = "cuda"
    args.data_location = "/kaggle/working/datasets"

    os.makedirs(args.results_dir, exist_ok=True)

    result_file = os.path.join(args.results_dir, "after_scaling_all_results.json")
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            existing_results = json.load(f)
        evaluated_datasets = {entry["dataset"] for entry in existing_results}
    else:
        existing_results = []
        evaluated_datasets = set()

    alpha_star = 0.30
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    for dataset_name in datasets:
        if dataset_name in evaluated_datasets:
            print(f"⏭️ Skipping {dataset_name} — already evaluated.")
            continue

        print(f"🔍 Evaluating after-scaling model for {dataset_name}")
        args.data_location = resolve_dataset_path(args, dataset_name)

        task_vector = NonLinearTaskVector(
            pretrained_checkpoint=os.path.join(args.checkpoints_path, "pretrained.pt"),
            finetuned_checkpoint=os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
        )

        scaled_vector = task_vector * alpha_star
        encoder = scaled_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

        head = get_classification_head(args, f"{dataset_name}Val").cuda()
        model = ImageClassifier(encoder, head).cuda()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(f"{dataset_name}Val", preprocess, args.data_location, args.batch_size)
        train_loader = dataset.train_loader
        test_loader = dataset.test_loader

        def eval_model(model, loader):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in loader:
                    x, y = batch[0].cuda(), batch[1].cuda()
                    out = model(x)
                    _, pred = out.max(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            return correct / total

        train_acc = eval_model(model, train_loader)
        test_acc = eval_model(model, test_loader)
        fim_log_trace = compute_fim_log_trace(args, model, dataset_name)

        print(f"✅ {dataset_name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, FIM={fim_log_trace:.4f}")

        # Save after each dataset
        result = {
            "dataset": dataset_name,
            "alpha": alpha_star,
            "scaled_train_accuracy": train_acc,
            "scaled_test_accuracy": test_acc,
            "fim_log_trace": fim_log_trace
        }
        existing_results.append(result)
        with open(result_file, "w") as f:
            json.dump(existing_results, f, indent=4)
        print(f"💾 Saved result for {dataset_name}")

    print("✅ All after-scaling results completed and saved!")

if __name__ == "__main__":
    evaluate_scaled_model()
