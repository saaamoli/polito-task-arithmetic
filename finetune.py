import os
import time
import torch
import sys

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

def fine_tune_on_dataset(args, dataset_name, num_epochs):
    print(f"\n==== Fine-tuning on {dataset_name} for {num_epochs} epochs ====\n")

    checkpoint_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint for {dataset_name} already exists at {checkpoint_path}. Skipping...")
        return

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset_path = resolve_dataset_path(args, dataset_name)
    args.data_location = base_dataset_path

    dataset = get_dataset(f"{dataset_name}Val", preprocess=preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    encoder = ImageEncoder(args).cuda()

    # ✅ Check if the classification head exists
    head_path = os.path.join(args.save, f"head_{dataset_name}Val.pt")
    if not os.path.exists(head_path):
        print(f"⚠️ Classification head for {dataset_name} not found. Generating one...")
        head = get_classification_head(args, dataset_name).cuda()  # Auto-generate head
        head.save(head_path)  # Save for future runs
        print(f"✅ Generated and saved classification head at {head_path}")
    else:
        print(f"✅ Loading existing classification head for {dataset_name} from {head_path}")
        head = torch.load(head_path).cuda()

    model = ImageClassifier(encoder, head).cuda()

    # ✅ Freeze the classification head
    for param in model.classification_head.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, early_stop_counter = 5, 0

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

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
                break

    save_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(save_path)
    print(f"✅ Fine-tuned model saved to {save_path}")

if __name__ == "__main__":
    args = parse_arguments()
    args.save = "/kaggle/working/checkpoints_updated"
    args.lr = 1e-4
    args.batch_size = 32

    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    for dataset_name, num_epochs in dataset_epochs.items():
        fine_tune_on_dataset(args, dataset_name, num_epochs)
