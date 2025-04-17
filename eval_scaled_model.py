
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
    args.results_dir = "/kaggle/working/results_after_scaling"
    args.save = args.results_dir
    args.device = "cuda"

    os.makedirs(args.results_dir, exist_ok=True)

    alpha_star = 0.3  # Your best alpha found in task addition
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    results = []

    for dataset_name in datasets:
        print(f"🔍 Evaluating after-scaling model for {dataset_name}")
        dataset_path = resolve_dataset_path(args, dataset_name)
        args.data_location = dataset_path

        # Load task vector
        task_vector = NonLinearTaskVector(
            pretrained_checkpoint=os.path.join(args.checkpoints_path, "pretrained.pt"),
            finetuned_checkpoint=os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
        )

        # Apply θ₀ + α⋆ * τₜ
        encoder = task_vector.apply_to(
            os.path.join(args.checkpoints_path, "pretrained.pt"),
            scale=alpha_star
        )

        # Load classification head
        head = get_classification_head(args, f"{dataset_name}Val").cuda()
        model = ImageClassifier(encoder, head).cuda()

        # Load dataset
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        train_loader = dataset.train_loader
        test_loader = dataset.test_loader

        # Evaluate
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

        results.append({
            "dataset": dataset_name,
            "alpha": alpha_star,
            "scaled_train_accuracy": train_acc,
            "scaled_test_accuracy": test_acc,
            "fim_log_trace": fim_log_trace
        })

    with open(os.path.join(args.results_dir, "after_scaling_all_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("✅ All after-scaling results saved!")

if __name__ == "__main__":
    evaluate_scaled_model()
