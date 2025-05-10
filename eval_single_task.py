import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize, get_dataloader
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms
from utils import train_diag_fim_logtr


# Load hyperparameters from hyperparams.json
hyperparams_path = '/kaggle/working/polito-task-arithmetic/hyperparams.json'
if not os.path.exists(hyperparams_path):
    raise FileNotFoundError(f"Hyperparameter configuration file not found at {hyperparams_path}")

with open(hyperparams_path, "r") as f:
    baseline_hyperparams = json.load(f)

def load_finetuned_model(args, dataset_name):
    encoder_checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    head_path = os.path.join(args.checkpoints_path, f"head_{dataset_name}Val.pt")

    if not os.path.exists(encoder_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_checkpoint_path}")

    encoder = ImageEncoder.load(args.model, encoder_checkpoint_path).cuda()

    if os.path.exists(head_path):
        print(f"✅ Loading existing classification head for {dataset_name} from {head_path}")
        head = torch.load(head_path).cuda()
    else:
        print(f"⚠️ Classification head for {dataset_name} not found. Generating one...")
        head = get_classification_head(args, dataset_name).cuda()
        os.makedirs(os.path.dirname(head_path), exist_ok=True)
        head.save(head_path)
        print(f"✅ Generated and saved classification head at {head_path}")

    return ImageClassifier(encoder, head).cuda()

def resolve_dataset_path(args, dataset_name):
    base_path = args.data_location
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "mnist":
        return os.path.join(base_path, "MNIST", "raw")
    elif dataset_name_lower == "gtsrb":
        return os.path.join(base_path, "gtsrb")
    elif dataset_name_lower == "svhn":
        return os.path.join(base_path, "svhn")
    else:
        return base_path

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
        print(f"✅ Results for {dataset_name} already exist at {save_path}. Skipping evaluation...")
        return

    original_data_location = args.data_location
    dataset_path = resolve_dataset_path(args, dataset_name)
    args.data_location = dataset_path

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    use_val_split = dataset_name in ["MNIST", "GTSRB", "SVHN", "EuroSAT", "RESISC45", "DTD"]

    if use_val_split:
        val_dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        train_loader = get_dataloader(val_dataset, is_train=True, args=args)
        val_loader = get_dataloader(val_dataset, is_train=False, args=args)
    else:
        base_dataset = get_dataset(dataset_name, preprocess, dataset_path, args.batch_size)
        train_loader = base_dataset.train_loader
        val_loader = base_dataset.test_loader

    test_dataset = get_dataset(dataset_name, preprocess, dataset_path, args.batch_size)
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)

    model = load_finetuned_model(args, dataset_name)
    train_acc = evaluate_model(model, train_loader)
    val_acc = evaluate_model(model, val_loader)
    test_acc = evaluate_model(model, test_loader)

    print(f"✅ Train Accuracy for {dataset_name}: {train_acc:.4f}")
    print(f"✅ Validation Accuracy for {dataset_name}: {val_acc:.4f}")
    print(f"✅ Test Accuracy for {dataset_name}: {test_acc:.4f}")

    fim_dataset = get_dataset(dataset_name, preprocess, resolve_dataset_path(args, dataset_name), args.batch_size)
    fim_loader = get_dataloader(fim_dataset, is_train=True, args=args)

    criterion = torch.nn.CrossEntropyLoss()
    fim_log_trace = train_diag_fim_logtr(args, model, dataset_name)
    print(f"📊 Log Tr[FIM] for {dataset_name}: {fim_log_trace:.4f}")

    results = {
        "dataset": dataset_name,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "fim_log_trace": fim_log_trace
    }
    save_results(results, save_path)
    args.data_location = original_data_location

def main():
    args = parse_arguments()

    # 🔁 Dynamic path setup based on --exp_name
    if args.save is None:
        if args.exp_name is not None:
            args.save = f"/kaggle/working/checkpoints_{args.exp_name}"
        else:
            args.save = "/kaggle/working/checkpoints_default"

    args.checkpoints_path = args.save
    args.results_dir = args.save.replace("checkpoints", "results")
    args.data_location = "/kaggle/working/datasets"

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    for dataset_name in datasets:
        print(f"\n--- Evaluating {dataset_name} ---")
        args.batch_size = baseline_hyperparams[dataset_name]["batch_size"]
        print(f"Using batch size {args.batch_size} for {dataset_name}")
        evaluate_and_save(args, dataset_name)

if __name__ == "__main__":
    main()
