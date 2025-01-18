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
from task_vectors import NonLinearTaskVector

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def save_pretrained_model(args):
    """Save the pre-trained encoder if not already saved."""
    save_path = os.path.join(args.checkpoints_path, "pretrained.pt")
    if not os.path.exists(save_path):
        print("🔄 Saving the pre-trained model...")
        encoder = ImageEncoder(args).cuda()
        encoder.save(save_path)
        print(f"✅ Pre-trained model saved at {save_path}")
    else:
        print(f"✅ Pre-trained model already exists at {save_path}")


def load_task_vector(args, dataset_name):
    """Load the task vector for a dataset using pre-trained and fine-tuned models."""
    pretrained_checkpoint = os.path.join(args.checkpoints_path, "pretrained.pt")
    finetuned_checkpoint = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")

    if not os.path.exists(pretrained_checkpoint):
        raise FileNotFoundError(f"Pre-trained checkpoint not found at {pretrained_checkpoint}")
    if not os.path.exists(finetuned_checkpoint):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found at {finetuned_checkpoint}")

    print(f"🔄 Loading task vector for {dataset_name}")
    return NonLinearTaskVector(pretrained_checkpoint=pretrained_checkpoint, finetuned_checkpoint=finetuned_checkpoint)


def resolve_dataset_path(args, dataset_name):
    """Resolve dataset path."""
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
    """Evaluate model accuracy."""
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
    """Compute normalized accuracy."""
    return np.mean([va / ba if ba != 0 else 0 for va, ba in zip(val_accuracies, best_accuracies)])


def evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies):
    """Evaluate the model for a specific alpha value on the validation set."""
    print(f"\n🔍 Evaluating alpha = {alpha:.2f}")

    # ✅ Combine task vectors using the current alpha
    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    # ✅ Apply the combined task vector to the pre-trained encoder
    blended_encoder = combined_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

    val_accuracies = []

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)

        # ✅ Apply grayscale-to-RGB conversion for MNIST
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        val_loader = dataset.train_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(blended_encoder, head).cuda()

        acc = evaluate_model(model, val_loader)
        val_accuracies.append(acc)
        print(f"📊 Validation Accuracy for {dataset_name}: {acc:.4f}")

    # ✅ Compute and return the average normalized accuracy
    avg_norm_acc = compute_average_normalized_accuracy(val_accuracies, best_accuracies)
    print(f"📈 Average Normalized Accuracy at alpha {alpha:.2f}: {avg_norm_acc:.4f}")
    return avg_norm_acc, val_accuracies

def evaluate_on_test(args, encoder, task_vectors, datasets, alpha):
    """Evaluate the model on test datasets using the best alpha and compute the average absolute accuracy."""
    print(f"\n🧪 Evaluating on Test Datasets with α = {alpha:.2f}")

    # ✅ Combine task vectors for the best alpha
    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    blended_encoder = combined_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

    test_accuracies = []

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)

        dataset = get_dataset(f"{dataset_name}Test", None, dataset_path, args.batch_size)
        test_loader = dataset.test_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(blended_encoder, head).cuda()

        acc = evaluate_model(model, test_loader)
        test_accuracies.append(acc)
        print(f"✅ Test Accuracy for {dataset_name}: {acc:.4f}")

    # ✅ Compute and print the average absolute accuracy (Equation 1)
    avg_absolute_acc = np.mean(test_accuracies)
    print(f"\n📊 **Average Absolute Accuracy on Test Sets**: {avg_absolute_acc:.4f}")
    return avg_absolute_acc


def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_updated"
    args.data_location = "/kaggle/working/datasets"
    args.results_dir = "/kaggle/working/results"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # ✅ Ensure pretrained model is saved
    save_pretrained_model(args)

    # ✅ Load task vectors
    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    # ✅ Load best validation accuracies
    best_accuracies = [json.load(open(os.path.join(args.results_dir, f"{ds}_results.json")))['validation_accuracy'] for ds in datasets]

    # 🔎 Search for the best alpha
    best_alpha, best_avg_norm_acc = 0, 0
    for alpha in np.arange(0.0, 1.05, 0.05):
        avg_norm_acc, _ = evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc, best_alpha = avg_norm_acc, alpha

    print(f"🏆 Best Alpha (α★): {best_alpha:.2f} with Avg Normalized Accuracy: {best_avg_norm_acc:.4f}")
   
    # ✅ Evaluate on test datasets using the best alpha
    evaluate_on_test(args, None, task_vectors, datasets, best_alpha)

if __name__ == "__main__":
    main()
