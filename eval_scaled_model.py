import os
import torch
import json
import argparse
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from datasets.registry import get_dataset
from utils import train_diag_fim_logtr as compute_fim_log_trace
from torchvision import transforms
from task_vectors import NonLinearTaskVector
from args import parse_arguments


def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            else:
                inputs, labels = batch[0].cuda(), batch[1].cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def resolve_dataset_path(base_path, dataset_name):
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "dtd":
        return os.path.join(base_path, "dtd")
    elif dataset_name_lower == "eurosat":
        return os.path.join(base_path, "eurosat")
    elif dataset_name_lower == "mnist":
        return os.path.join(base_path, "MNIST", "raw")
    elif dataset_name_lower == "gtsrb":
        return os.path.join(base_path, "gtsrb")
    elif dataset_name_lower == "resisc45":
        return os.path.join(base_path, "resisc45")
    elif dataset_name_lower == "svhn":
        return os.path.join(base_path, "svhn")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--pretrained_path", type=str, default="/kaggle/working/checkpoints_batchsize/pretrained.pt")
    parser.add_argument("--checkpoints_dir", type=str, default="/kaggle/working/checkpoints_batchsize")
    parser.add_argument("--results_dir", type=str, default="/kaggle/working/results_after_scaling")
    parser.add_argument("--data_location", type=str, default="/kaggle/working/datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Add missing attributes to match other scripts
    args.model = "ViT-B-32__pretrained__openai"
    args.save = args.results_dir
    args.device = "cuda"
    args.cache_dir = None
    args.openclip_cachedir = None

    os.makedirs(args.results_dir, exist_ok=True)

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    all_results = []

    for dataset_name in datasets:
        print(f"\n--- Evaluating {dataset_name} ---")
        dataset_path = resolve_dataset_path(args.data_location, dataset_name)
        finetuned_path = os.path.join(args.checkpoints_dir, f"{dataset_name}_finetuned.pt")
        task_vector = NonLinearTaskVector(args.pretrained_path, finetuned_path)
        encoder = task_vector.apply_to(args.pretrained_path, scaling_coef=args.alpha).cuda()

        head = get_classification_head(args, f"{dataset_name}Val").cuda()
        model = ImageClassifier(encoder, head).cuda()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, batch_size=args.batch_size)
        train_loader = dataset.train_loader
        test_dataset = get_dataset(dataset_name, preprocess, dataset_path, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        train_acc = evaluate_model(model, train_loader)
        test_acc = evaluate_model(model, test_loader)
        fim_log_trace = compute_fim_log_trace(args, model, dataset_name, samples_nr=2000)

        result = {
            "dataset": dataset_name,
            "alpha": args.alpha,
            "scaled_train_accuracy": train_acc,
            "scaled_test_accuracy": test_acc,
            "fim_log_trace": fim_log_trace
        }

        all_results.append(result)
        print(json.dumps(result, indent=4))

    # Save all results to one file
    results_path = os.path.join(args.results_dir, "after_scaling_all_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✅ All results saved to {results_path}")


if __name__ == "__main__":
    main()
