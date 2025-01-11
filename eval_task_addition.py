import os
import json
import torch
import numpy as np
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms
import copy


from heads import ClassificationHead  # ✅ Import the correct class
import torch

def load_task_vector(args, dataset_name):
    """Load classification head (task vector) for a dataset."""
    head_path = os.path.join(args.results_dir, f"head_{dataset_name}Val.pt")
    
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Task vector not found: {head_path}")

    # ✅ Allowlist both ClassificationHead and set
    torch.serialization.add_safe_globals({ClassificationHead, set})

    # ✅ Load the classification head safely
    return torch.load(head_path, weights_only=True).cuda()







def evaluate_model(model, dataloader):
    """Evaluate model accuracy on provided data loader."""
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


def compute_average_normalized_accuracy(val_accuracies, best_accuracies):
    """Compute average normalized accuracy."""
    normalized_accs = [va / ba if ba != 0 else 0 for va, ba in zip(val_accuracies, best_accuracies)]
    return np.mean(normalized_accs)


def combine_task_vectors(task_vectors, alpha):
    """Combine task vectors with scaling by alpha, handling shape mismatches."""
    combined_vector = copy.deepcopy(task_vectors[0])

    for vec in task_vectors[1:]:
        for param_combined, param_vec in zip(combined_vector.parameters(), vec.parameters()):
            if param_combined.data.shape == param_vec.data.shape:
                param_combined.data += param_vec.data
            else:
                # Optional: Comment out if you don't want the warning
                print(f"⚠️ Skipping incompatible parameters: {param_combined.shape} vs {param_vec.shape}")

    # Scale the combined vector by alpha
    for param in combined_vector.parameters():
        param.data *= alpha

    return combined_vector


def evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies):
    """Evaluate the model with a specific alpha on all validation datasets."""
    val_accuracies = []

    for dataset_name in datasets:
        dataset_path = os.path.join(args.data_location, dataset_name.lower())
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        val_loader = dataset.train_loader

        combined_task_vector = combine_task_vectors(task_vectors, alpha)

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        acc = evaluate_model(model, val_loader)
        val_accuracies.append(acc)

    avg_norm_acc = compute_average_normalized_accuracy(val_accuracies, best_accuracies)
    return avg_norm_acc, val_accuracies


def main():
    args = parse_arguments()

    args.checkpoints_path = "/kaggle/working/checkpoints"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # ✅ Load encoder
    encoder = ImageEncoder(args).cuda()

    # ✅ Load all task vectors
    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    # ✅ Load best accuracies from single-task evaluations
    best_accuracies = []
    for dataset in datasets:
        result_path = os.path.join(args.results_dir, f"{dataset}_results.json")
        with open(result_path, 'r') as file:
            data = json.load(file)
            best_accuracies.append(data['validation_accuracy'])

    # ✅ Search for the best alpha
    best_alpha = 0
    best_avg_norm_acc = 0

    for alpha in np.arange(0.0, 1.05, 0.05):
        avg_norm_acc, _ = evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc = avg_norm_acc
            best_alpha = alpha

    print(f"✅ Best alpha (α★): {best_alpha} with Avg Normalized Accuracy: {best_avg_norm_acc:.4f}")

    # ✅ Evaluate on test sets using α★
    test_accuracies = []
    for dataset_name in datasets:
        dataset_path = os.path.join(args.data_location, dataset_name.lower())
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        test_loader = dataset.test_loader

        combined_task_vector = combine_task_vectors(task_vectors, best_alpha)
        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        acc = evaluate_model(model, test_loader)
        test_accuracies.append(acc)

    avg_abs_acc = np.mean(test_accuracies)
    avg_norm_acc = compute_average_normalized_accuracy(test_accuracies, best_accuracies)

    # ✅ Save final results
    results = {
        "best_alpha": best_alpha,
        "average_absolute_accuracy": avg_abs_acc,
        "average_normalized_accuracy": avg_norm_acc
    }

    save_path = os.path.join(args.results_dir, "multi_task_results.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Multi-task evaluation completed. Results saved to {save_path}")


if __name__ == "__main__":
    main()
