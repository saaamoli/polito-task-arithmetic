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
    """Load classification head (task vector) for a dataset."""
    head_path = os.path.join(args.results_dir, f"head_{dataset_name}Val.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"❌ Task vector not found: {head_path}")

    torch.serialization.add_safe_globals({ClassificationHead, set})
    print(f"🔄 Loading classification head for {dataset_name} from {head_path}")
    return torch.load(head_path, map_location="cuda")


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
    accuracy = correct / total
    return accuracy


def combine_task_vectors(task_vectors, alpha):
    """Combine task vectors with scaling by alpha, ensuring strict shape consistency."""
    combined_vector = copy.deepcopy(task_vectors[0])

    for vec_index, vec in enumerate(task_vectors[1:], start=1):
        for param_combined, param_vec in zip(combined_vector.parameters(), vec.parameters()):
            if param_combined.data.shape == param_vec.data.shape:
                param_combined.data += param_vec.data
            else:
                raise ValueError(
                    f"❌ Shape mismatch while combining task vectors at index {vec_index}: "
                    f"{param_combined.shape} vs {param_vec.shape}"
                )

    # Apply scaling by alpha
    for param in combined_vector.parameters():
        param.data *= alpha

    print(f"🟢 Successfully combined task vectors with alpha = {alpha}")
    return combined_vector


def evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies):
    """Evaluate the model with a specific alpha on all datasets."""
    print(f"\n🔍 Evaluating alpha = {alpha:.2f}")
    val_accuracies = []

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)
        print(f"📊 Evaluating dataset: {dataset_name}")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        val_loader = dataset.train_loader

        combined_task_vector = combine_task_vectors(task_vectors, alpha)
        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        acc = evaluate_model(model, val_loader)
        print(f"✅ Accuracy on {dataset_name} at alpha {alpha:.2f}: {acc:.4f}")
        val_accuracies.append(acc)

    avg_norm_acc = np.mean(val_accuracies)
    print(f"📈 Average accuracy at alpha {alpha:.2f}: {avg_norm_acc:.4f}")
    return avg_norm_acc, val_accuracies


def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    print("🚀 Starting multi-task evaluation...")
    encoder = ImageEncoder(args).cuda()
    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    best_accuracies = []
    for dataset in datasets:
        result_path = os.path.join(args.results_dir, f"{dataset}_results.json")
        with open(result_path, 'r') as file:
            data = json.load(file)
            best_accuracies.append(data['validation_accuracy'])

    best_alpha, best_avg_norm_acc = 0, 0

    for alpha in np.arange(0.0, 1.05, 0.05):
        avg_norm_acc, _ = evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc, best_alpha = avg_norm_acc, alpha

    print(f"\n🏆 Best alpha (α★): {best_alpha:.2f} with Avg Normalized Accuracy: {best_avg_norm_acc:.4f}")
    print("✅ Multi-task evaluation completed.")


if __name__ == "__main__":
    main()
