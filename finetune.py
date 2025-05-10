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


def resolve_dataset_path(args, dataset_name):
    base_path = args.data_location
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "dtd":
        return os.path.join(base_path)
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

    # ✅ Temporarily adjust dataset path
    original_data_location = args.data_location
    args.data_location = resolve_dataset_path(args, dataset_name)

    checkpoint_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint for {dataset_name} already exists at {checkpoint_path}. Skipping...")
        args.data_location = original_data_location
        return

    encoder = ImageEncoder(args).to(args.device)
    preprocess = encoder.train_preprocess

    dataset = get_dataset(f"{dataset_name}Val", preprocess=preprocess, location=args.data_location, batch_size=batch_size, num_workers=2)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    head = get_classification_head(args, dataset_name).to(args.device)
    model = ImageClassifier(encoder, head).to(args.device)

    # Freeze the classification head
    model.freeze_head()


    optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    results = {
        "dataset": dataset_name,
        "lr": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "epochs": []
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].to(args.device), data["labels"].to(args.device)

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
                inputs, labels = data["images"].to(args.device), data["labels"].to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
        results["epochs"].append({
            "epoch": epoch+1,
            "train_loss": epoch_loss / len(train_loader),
            "val_loss": avg_val_loss,
            "val_acc": val_accuracy
        })

    # Save model and results
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_finetuned.pt"))
    print(f"✅ Fine-tuned model saved to {os.path.join(args.save, f'{dataset_name}_finetuned.pt')}")

    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(results) + "\n")

    # ✅ Restore original path
    args.data_location = original_data_location


if __name__ == "__main__":
    args = parse_arguments()
    # Set save path dynamically if not provided
    if args.save is None:
    if args.exp_name is not None:
        args.save = f"/kaggle/working/checkpoints_{args.exp_name}"
    else:
        args.save = "/kaggle/working/checkpoints_default"

    args.data_location = "/kaggle/working/datasets"

    # ✅ Reproducibility (optional)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load hyperparameters
    hyperparams_path = os.path.join('/kaggle/working/polito-task-arithmetic', 'hyperparams.json')
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"Hyperparameter configuration file not found at {hyperparams_path}")
    
    with open(hyperparams_path, "r") as f:
        baseline_hyperparams = json.load(f)

    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    log_path = "/kaggle/working/weight_results.json"

    for dataset_name, num_epochs in dataset_epochs.items():
        hyperparams = baseline_hyperparams[dataset_name]
        fine_tune_on_dataset(
            args,
            dataset_name,
            num_epochs,
            hyperparams["learning_rate"],
            hyperparams["batch_size"],
            hyperparams["weight_decay"],
            log_path,
        )
