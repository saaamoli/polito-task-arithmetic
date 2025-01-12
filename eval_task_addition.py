import os
import json
import torch
import numpy as np
import warnings
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms
from task_vectors import NonLinearTaskVector  # ✅ Import task vector handling

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_task_vector(args, dataset_name):
    checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Task vector not found: {checkpoint_path}")
    
    print(f"🔄 Loading task vector for {dataset_name}")
    return NonLinearTaskVector(pretrained_checkpoint=None, finetuned_checkpoint=checkpoint_path)


def resolve_dataset_path(args, dataset_name):
    base_path = args.data_location
    dataset_paths = {
        "dtd": os.path.join(base_path, "dtd"),
        "eurosat": base_path,
        "mnist": os.path.join(base_path, "MNIST", "raw"),
        "gtsrb": os.path.join(base_path, "gtsrb"),
        "resisc45": base_path,
        "svhn": os.path.join(base_path, "svhn"),
    }
    return dataset_paths.get(dataset_name.lower(), None)


def evaluate_model(model, dataloader):
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


def compute_average_normalized_accuracy(val_accuracies, best_accuracies):
    normalized_accs = [va / ba if ba != 0 else 0 for va, ba in zip(val_accuracies, best_accuracies)]
    return np.mean(normalized_accs)


def evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies):
    val_accuracies = []
    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    encoder = combined_vector.apply_to(encoder)

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        val_loader = dataset.train_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        acc = evaluate_model(model, val_loader)
        val_accuracies.append(acc)

    return compute_average_normalized_accuracy(val_accuracies, best_accuracies), val_accuracies


def evaluate_on_test(args, encoder, task_vectors, datasets, alpha):
    test_accuracies = []
    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    encoder = combined_vector.apply_to(encoder)

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)
        dataset = get_dataset(f"{dataset_name}Test", None, dataset_path, args.batch_size)
        test_loader = dataset.test_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        acc = evaluate_model(model, test_loader)
        test_accuracies.append(acc)
        print(f"✅ Test Accuracy on {dataset_name}: {acc:.4f}")

    avg_test_acc = np.mean(test_accuracies)
    print(f"📊 Average Absolute Accuracy on Test Sets: {avg_test_acc:.4f}")


def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_updated"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    encoder = ImageEncoder(args).cuda()
    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    best_accuracies = [json.load(open(os.path.join(args.results_dir, f"{ds}_results.json")))['validation_accuracy'] for ds in datasets]

    best_alpha, best_avg_norm_acc = 0, 0
    for alpha in np.arange(0.0, 1.05, 0.05):
        avg_norm_acc, _ = evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc, best_alpha = avg_norm_acc, alpha

    evaluate_on_test(args, encoder, task_vectors, datasets, best_alpha)


if __name__ == "__main__":
    main()
