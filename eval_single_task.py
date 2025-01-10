import os
import json
import torch
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms


def load_finetuned_model(args, dataset_name):
    encoder_checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")

    if not os.path.exists(encoder_checkpoint_path):
        print(f"⚠️ Checkpoint for {dataset_name} not found at {encoder_checkpoint_path}. Skipping...")
        return None

    print(f"✅ Loading checkpoint for {dataset_name} from {encoder_checkpoint_path}")
    encoder = torch.load(encoder_checkpoint_path, map_location="cuda")
    head = get_classification_head(args, dataset_name).cuda()
    model = ImageClassifier(encoder, head).cuda()

    return model


def resolve_dataset_path(args, dataset_name):
    """
    Correct dataset paths for each dataset.
    """
    base_path = args.data_location
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "dtd":
        return os.path.join(base_path, "dtd")
    elif dataset_name_lower == "eurosat":
        return os.path.join(base_path, "EuroSAT_splits")  # ✅ Corrected EuroSAT path
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


def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            elif isinstance(batch, (tuple, list)):
                inputs, labels = batch[0].cuda(), batch[1].cuda()
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to {save_path}")


def evaluate_and_save(args, dataset_name):
    save_path = os.path.join(args.results_dir, f"{dataset_name}_results.json")
    if os.path.exists(save_path):
        print(f"✅ Results for {dataset_name} already exist. Skipping...")
        return

    dataset_path = resolve_dataset_path(args, dataset_name)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
    val_loader = dataset.train_loader
    test_loader = dataset.test_loader

    model = load_finetuned_model(args, dataset_name)

    if model is None:
        return

    val_acc = evaluate_model(model, val_loader)
    test_acc = evaluate_model(model, test_loader)

    results = {
        "dataset": dataset_name,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc
    }

    save_results(results, save_path)


def main():
    args = parse_arguments()

    args.checkpoints_path = "/kaggle/working/checkpoints"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    for dataset_name in datasets:
        print(f"\n--- Evaluating {dataset_name} ---")
        evaluate_and_save(args, dataset_name)


if __name__ == "__main__":
    main()
