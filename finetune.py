import os
import time
import torch
import sys
import json

# Add project root to Python path
sys.path.append('/kaggle/working/polito-task-arithmetic')
print("Python Path:", sys.path)

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
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

def fine_tune_on_dataset(args, dataset_name, num_epochs, learning_rate, batch_size, weight_decay, log_path):
    print(f"\n==== Fine-tuning on {dataset_name} with LR={learning_rate}, Batch Size={batch_size}, WD={weight_decay} ====\n")

    checkpoint_path = os.path.join(args.save, f"{dataset_name}_finetuned_lr{learning_rate}_bs{batch_size}_wd{weight_decay}.pt")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint for {dataset_name} already exists at {checkpoint_path}. Skipping...")
        return

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset_path = resolve_dataset_path(args, dataset_name)
    args.data_location = base_dataset_path

    dataset = get_dataset(f"{dataset_name}Val", preprocess=preprocess, location=args.data_location, batch_size=batch_size, num_workers=2)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    encoder = ImageEncoder(args).cuda()
    head = get_classification_head(args, dataset_name).cuda()
    model = ImageClassifier(encoder, head).cuda()

    # ✅ Freeze the classification head
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

    save_path = os.path.join(args.save, f"{dataset_name}_finetuned_lr{learning_rate}_bs{batch_size}_wd{weight_decay}.pt")
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(save_path)
    print(f"✅ Fine-tuned model saved to {save_path}")

    # Save results to JSON
    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(results) + "\n")

if __name__ == "__main__":
    args = parse_arguments()
    args.save = "/kaggle/working/checkpoints_updated"
    args.data_location = "/kaggle/working/datasets"

    # Grid Search Parameters
    batch_sizes = [8, 16, 32, 64]
    learning_rates = [5e-4, 5e-5, 1e-5]
    weight_decays = [0.001, 0.01, 0.1]

    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    log_path = "/kaggle/working/grid_search_results.json"

    for dataset_name, num_epochs in dataset_epochs.items():
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for wd in weight_decays:
                    fine_tune_on_dataset(args, dataset_name, num_epochs, lr, batch_size, wd, log_path)
