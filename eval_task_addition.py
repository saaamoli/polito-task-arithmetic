import os
import json
import torch
import numpy as np
import warnings
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head, ClassificationHead
from args import parse_arguments
from torchvision import transforms
import copy

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_task_vector(args, dataset_name):
    """
    Load the encoder task vector for the given dataset.
    """
    checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Task vector (encoder) not found: {checkpoint_path}")
    
    print(f"🔄 Loading encoder task vector for {dataset_name} from {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cuda")


def resolve_dataset_path(args, dataset_name):
    """Resolves dataset path based on dataset name."""
    base_path = args.data_location
    dataset_paths = {
        "dtd": os.path.join(base_path, "dtd"),
        "eurosat": base_path,
        "mnist": os.path.join(base_path, "MNIST", "raw"),
        "gtsrb": os.path.join(base_path, "gtsrb"),
        "resisc45": base_path,
        "svhn": os.path.join(base_path, "svhn"),
    }
    if dataset_name.lower() in dataset_paths:
        return dataset_paths[dataset_name.lower()]
    else:
        raise ValueError(f"❌ Unknown dataset: {dataset_name}")


def evaluate_model(model, dataloader):
    """Evaluate model accuracy on the given dataloader."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_on_split(args, encoder, datasets, alpha, split_type="val"):
    """Evaluate the model on validation or test split."""
    accuracies = []

    # ✅ Correct function call to combine task vectors
    combined_vector = combine_task_vectors(
        [load_task_vector(args, ds) for ds in datasets], alpha
    )

    # ✅ Proper blending with encoder
    blend_with_encoder(encoder, combined_vector, alpha)

    for dataset_name in datasets:
        print(f"📊 Evaluating {split_type} set of {dataset_name} at alpha {alpha}")
        dataset_path = resolve_dataset_path(args, dataset_name)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        dataloader = dataset.train_loader if split_type == "val" else dataset.test_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        acc = evaluate_model(model, dataloader)
        print(f"✅ {split_type.capitalize()} Accuracy on {dataset_name}: {acc:.4f}")
        accuracies.append(acc)

    return accuracies



def compute_average_normalized_accuracy(val_accuracies, best_accuracies):
    """Compute average normalized accuracy based on Equation 2."""
    normalized_accs = [va / ba if ba != 0 else 0 for va, ba in zip(val_accuracies, best_accuracies)]
    return np.mean(normalized_accs)


def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_updated"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    encoder = ImageEncoder(args).cuda()

    # Load best validation accuracies
    best_accuracies = []
    for dataset in datasets:
        result_path = os.path.join(args.results_dir, f"{dataset}_results.json")
        with open(result_path, 'r') as file:
            data = json.load(file)
            best_accuracies.append(data['validation_accuracy'])

    # 🔍 Find best alpha
    best_alpha, best_avg_norm_acc = 0, 0
    for alpha in np.arange(0.0, 1.05, 0.05):
        val_accuracies = evaluate_on_split(args, encoder, datasets, alpha, split_type="val")
        avg_norm_acc = compute_average_normalized_accuracy(val_accuracies, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc, best_alpha = avg_norm_acc, alpha

    print(f"\n🏆 Best alpha (α★): {best_alpha:.2f} with Avg Normalized Accuracy: {best_avg_norm_acc:.4f}")

    # ✅ Evaluate on test set with best α★
    test_accuracies = evaluate_on_split(args, encoder, datasets, best_alpha, split_type="test")
    avg_abs_acc = np.mean(test_accuracies)
    avg_norm_acc = compute_average_normalized_accuracy(test_accuracies, best_accuracies)

    # 💾 Save results
    final_results = {
        "best_alpha": best_alpha,
        "average_absolute_accuracy": avg_abs_acc,
        "average_normalized_accuracy": avg_norm_acc,
        "test_accuracies": {datasets[i]: acc for i, acc in enumerate(test_accuracies)}
    }

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "multi_task_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"✅ Final multi-task results saved to {results_path}")


if __name__ == "__main__":
    main()

