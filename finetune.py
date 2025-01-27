import os
import time
import torch
import sys
import json
import argparse

# Add project root to Python path
sys.path.append('/kaggle/working/polito-task-arithmetic')
print("Python Path:", sys.path)

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
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


def fine_tune_on_dataset(dataset_name, num_epochs, learning_rate, batch_size, weight_decay, log_path, data_location, save_path):
    print(f"\n==== Fine-tuning on {dataset_name} with LR={learning_rate}, Batch Size={batch_size}, WD={weight_decay} ====\n")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset_path = resolve_dataset_path(data_location, dataset_name)

    if dataset_name.lower() == "dtd":
        train_dataset = get_dataset(
            f"{dataset_name}",
            preprocess=preprocess,
            location=base_dataset_path["train"],
            batch_size=batch_size,
            num_workers=2
        )
        val_dataset = get_dataset(
            f"{dataset_name}",
            preprocess=preprocess,
            location=base_dataset_path["val"],
            batch_size=batch_size,
            num_workers=2
        )
    else:
        dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=preprocess,
            location=base_dataset_path,
            batch_size=batch_size,
            num_workers=2
        )
        train_dataset = dataset
        val_dataset = dataset

    train_loader = get_dataloader(train_dataset, is_train=True)
    val_loader = get_dataloader(val_dataset, is_train=False)

    encoder = ImageEncoder().cuda()
    head = get_classification_head(dataset_name).cuda()
    model = ImageClassifier(encoder, head).cuda()

    # Freeze the classification head
    for param in model.classification_head.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    results = {"dataset": dataset_name, "lr": learning_rate, "batch_size": batch_size, "weight_decay": weight_decay, "epochs": []}

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

    # Save the model
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{dataset_name}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}.pt")
    torch.save(model.image_encoder.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # Save results to JSON
    with open(log_path, "a") as log_file:
        json.dump(results, log_file)
        log_file.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune models with different hyperparameters")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., DTD, EuroSAT, GTSRB, MNIST, RESISC45, SVHN)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for optimization")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay for optimization")
    parser.add_argument("--log_path", type=str, required=True, help="Path to save results log")
    args = parser.parse_args()

    fine_tune_on_dataset(
        args.dataset,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.log_path,
    )
