import os
import json
import torch
import numpy as np
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms


def load_task_vector(args, dataset_name):
    """Load classification head (task vector) for a dataset."""
    head_path = os.path.join(args.results_dir, f"head_{dataset_name}Val.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Task vector not found: {head_path}")
    return torch.load(head_path).cuda()


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
    normalized_accs = [va / ba for va, ba in zip(val_accuracies, best_accuracies)]
    return np.mean(normalized_accs)


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

       # Initialize combined_task_vector with the first task vector
        combined_task_vector = task_vectors[0].clone()
        
        # Add the remaining task vectors
        for vec in task_vectors[1:]:
            combined_task_vector += vec
        
        # Scale by alpha
        combined_task_vector *= alpha

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder + combined_task_vector, head).cuda()

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

    encoder = ImageEncoder(args).cuda()

    # Load all task vectors
    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    # Load best accuracies from single-task evaluations
    best_accuracies = []
    for dataset in datasets:
        result_path = os.path.join(args.results_dir, f"{dataset}_results.json")
        with open(result_path, 'r') as file:
            data = json.load(file)
            best_accuracies.append(data['validation_accuracy'])

    # Search for the best alpha
    best_alpha = 0
    best_avg_norm_acc = 0

    for alpha in np.arange(0.0, 1.05, 0.05):
        avg_norm_acc, _ = evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc = avg_norm_acc
            best_alpha = alpha

    print(f"✅ Best alpha (α★): {best_alpha} with Avg Normalized Accuracy: {best_avg_norm_acc:.4f}")

    # Evaluate on test sets using α★
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

        combined_task_vector = sum(task_vectors) * best_alpha
        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder + combined_task_vector, head).cuda()

        acc = evaluate_model(model, test_loader)
        test_accuracies.append(acc)

    avg_abs_acc = np.mean(test_accuracies)
    avg_norm_acc = compute_average_normalized_accuracy(test_accuracies, best_accuracies)

    # Save final results
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