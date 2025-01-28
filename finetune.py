import os
import sys
import json
import torch
from argparse import ArgumentParser
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from torchvision import transforms

# Add project root to Python path
sys.path.append('/kaggle/working/polito-task-arithmetic')
print("Python Path:", sys.path)

def parse_arguments():
    parser = ArgumentParser(description="Fine-tune image classifier on various datasets.")
    parser.add_argument("--data-location", type=str, required=True, help="Path to the dataset location.")
    parser.add_argument("--save", type=str, required=True, help="Path to save fine-tuned checkpoints.")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate for fine-tuning.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--weight-decay", type=float, required=True, help="Weight decay for optimizer.")
    return parser.parse_args()


def resolve_dataset_path(data_location, dataset_name):
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "dtd":
        return os.path.join(data_location, "dtd")
    elif dataset_name_lower == "eurosat":
        return data_location
    elif dataset_name_lower == "mnist":
        return os.path.join(data_location, "MNIST", "raw")
    elif dataset_name_lower == "gtsrb":
        return os.path.join(data_location, "gtsrb")
    elif dataset_name_lower == "resisc45":
        return data_location
    elif dataset_name_lower == "svhn":
        return os.path.join(data_location, "svhn")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def fine_tune_on_dataset(args, dataset_name, num_epochs, log_path):
    print(f"\n==== Fine-tuning on {dataset_name} with LR={args.learning_rate}, Batch Size={args.batch_size}, WD={args.weight_decay} ====\n")

    checkpoint_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint for {dataset_name} already exists at {checkpoint_path}. Skipping...")
        return

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_path = resolve_dataset_path(args.data_location, dataset_name)
    dataset = get_dataset(f"{dataset_name}Val", preprocess=preprocess, location=dataset_path, batch_size=args.batch_size, num_workers=2)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    # Handle dataset class names
    if hasattr(dataset, "classnames"):
        num_classes = len(dataset.classnames)
    elif hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    else:
        raise AttributeError(f"Dataset {dataset_name} does not have 'classnames' or 'classes' attributes.")

    # Ensure required attributes in args
    args.num_classes = num_classes
    args.image_size = 224  # Default image size

    encoder = ImageEncoder(args).cuda()
    head = get_classification_head(args, dataset_name).cuda()
    model = ImageClassifier(encoder, head).cuda()

    # Freeze the classification head
    for param in model.classification_head.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    results = {"dataset": dataset_name, "lr": args.learning_rate, "batch_size": args.batch_size, "weight_decay": args.weight_decay, "epochs": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].cuda(), data["labels"].cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                data = maybe_dictionarize(batch)
                inputs, labels = data["images"].cuda(), data["labels"].cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
        results["epochs"].append({"epoch": epoch+1, "train_loss": epoch_loss / len(train_loader), "val_loss": avg_val_loss, "val_acc": val_accuracy})

    save_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(save_path)
    print(f"✅ Fine-tuned model saved to {save_path}")

    # Save results to JSON
    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    args = parse_arguments()
    args.save = "/kaggle/working/checkpoints_baseline"  # New directory for baseline checkpoints
    args.data_location = "/kaggle/working/datasets"

    # Define datasets and epochs
    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    log_path = os.path.join(args.save, "baseline_results.json")

    for dataset_name, num_epochs in dataset_epochs.items():
        fine_tune_on_dataset(
            args,
            dataset_name,
            num_epochs,
            learning_rate=1e-4,
            batch_size=32,
            weight_decay=0.0,
            log_path=log_path,
        )
