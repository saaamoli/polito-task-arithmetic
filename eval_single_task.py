import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms

def load_finetuned_model(args, dataset_name):
    encoder_checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")

    if not os.path.exists(encoder_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_checkpoint_path}")

    encoder = torch.load(encoder_checkpoint_path).cuda()

    # Load or Generate the Classification Head
    head_path = os.path.join(args.results_dir, f"head_{dataset_name}Val.pt")
    if os.path.exists(head_path):
        print(f"✅ Loading existing classification head for {dataset_name} from {head_path}")
        head = torch.load(head_path).cuda()
    else:
        print(f"⚠️ Classification head for {dataset_name} not found. Generating one...")
        head = get_classification_head(args, dataset_name).cuda()
        head.save(head_path)
        print(f"✅ Generated and saved classification head at {head_path}")

    model = ImageClassifier(encoder, head).cuda()
    return model

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

def compute_fim_log_trace(model, dataloader, criterion, device):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    total_samples = 0
    max_samples = 2000  # Desired number of samples for FIM computation
    dataloader_iterator = iter(dataloader)  # Create an iterator for looping through the dataloader

    print(f"Starting FIM computation with dataset size: {len(dataloader.dataset)}")
    while total_samples < max_samples:
        try:
            batch = next(dataloader_iterator)  # Get the next batch
        except StopIteration:
            # Restart the iterator if we run out of batches
            dataloader_iterator = iter(dataloader)
            batch = next(dataloader_iterator)

        # Handle different batch formats
        if isinstance(batch, dict):  
            inputs, labels = batch['images'].to(device), batch['labels'].to(device)
        elif isinstance(batch, (list, tuple)):  
            inputs, labels = batch[0].to(device), batch[1].to(device)
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        # Reset gradients
        model.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)
        if not loss.requires_grad:
            raise ValueError("Loss does not require gradients. Check the computation graph.")

        # Backward pass
        loss.backward(retain_graph=True)

        # Accumulate FIM values
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim[name] += param.grad.pow(2)

        total_samples += inputs.size(0)
        if total_samples >= max_samples:
            print(f"Processed {total_samples} samples for FIM computation.")
            break

    # Compute the log-trace of FIM
    fim_trace = sum(fim_value.sum().item() for fim_value in fim.values())
    fim_log_trace = torch.log(torch.tensor(fim_trace / total_samples))  # Normalize by total_samples
    return fim_log_trace.item()


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

    dataset_path = resolve_dataset_path(args, dataset_name)
    args.data_location = dataset_path

    # Apply Grayscale to RGB conversion for MNIST
    if dataset_name.lower() == "mnist":
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
    train_loader = dataset.train_loader  # Use training set for FIM computation
    val_loader = dataset.train_loader  # Validation set for accuracy
    test_loader = dataset.test_loader  # Test set for accuracy

    model = load_finetuned_model(args, dataset_name)

    # Compute validation accuracy
    val_acc = evaluate_model(model, val_loader)

    # Compute test accuracy
    test_acc = evaluate_model(model, test_loader)

    # Compute FIM log-trace on the training set
    criterion = torch.nn.CrossEntropyLoss()
    fim_log_trace = compute_fim_log_trace(model, train_loader, criterion, device=args.device)

    results = {
        "dataset": dataset_name,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,  # Added test accuracy
        "fim_log_trace": fim_log_trace
    }

    save_results(results, save_path)

def main():
    args = parse_arguments()

    # Define necessary paths
    args.checkpoints_path = "/kaggle/working/checkpoints_updated"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.save = "/kaggle/working/checkpoints_updated"
    args.batch_size = 32

    # List of datasets to evaluate
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Evaluate and save results for each dataset
    for dataset_name in datasets:
        print(f"\n--- Evaluating {dataset_name} ---")
        evaluate_and_save(args, dataset_name)

if __name__ == "__main__":
    main()
