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
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def save_pretrained_model(args):
    save_path = os.path.join(args.checkpoints_path, "pretrained.pt")
    if os.path.exists(save_path):
        print(f"✅ Pre-trained model already exists at {save_path}. Skipping...")
        return False
    print("🔄 Saving the pre-trained model...")
    encoder = ImageEncoder(args).cuda()
    encoder.save(save_path)
    print(f"✅ Pre-trained model saved at {save_path}")
    return True

def load_task_vector(args, dataset_name):
    pretrained_checkpoint = os.path.join(args.checkpoints_path, "pretrained.pt")
    finetuned_checkpoint = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")

    if not os.path.exists(pretrained_checkpoint):
        raise FileNotFoundError(f"Pre-trained checkpoint not found at {pretrained_checkpoint}")
    if not os.path.exists(finetuned_checkpoint):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found at {finetuned_checkpoint}")

    print(f"🔄 Loading task vector for {dataset_name}")
    return NonLinearTaskVector(pretrained_checkpoint=pretrained_checkpoint, finetuned_checkpoint=finetuned_checkpoint)

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

def compute_fim_log_trace(model, dataloader, criterion, device):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    total_samples = 0
    max_samples = 2000
    dataloader_iterator = iter(dataloader)

    print(f"Starting FIM computation with dataset size: {len(dataloader.dataset)}")
    while total_samples < max_samples:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            batch = next(dataloader_iterator)

        inputs, labels = batch[0].to(device), batch[1].to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim[name] += param.grad.pow(2)

        total_samples += inputs.size(0)
        if total_samples >= max_samples:
            print(f"Processed {total_samples} samples for FIM computation.")
            break

    fim_trace = sum(fim_value.sum().item() for fim_value in fim.values())
    normalized_trace = max(fim_trace / (total_samples or 1), 1e-6)
    fim_log_trace = torch.log(torch.tensor(normalized_trace))

    print(f"Raw FIM Trace Sum: {fim_trace}")
    print(f"Normalized FIM Trace: {normalized_trace}, Log Trace: {fim_log_trace.item()}")
    return fim_log_trace.item()

def evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies):
    print(f"\n🔍 Evaluating alpha = {alpha:.2f}")

    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    blended_encoder = combined_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

    val_accuracies = []
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
        model = ImageClassifier(blended_encoder, head).cuda()

        acc = evaluate_model(model, val_loader)
        val_accuracies.append(acc)

    avg_norm_acc = np.mean([va / ba if ba != 0 else 0 for va, ba in zip(val_accuracies, best_accuracies)])
    return avg_norm_acc

def find_best_alpha(args, encoder, task_vectors, datasets, best_accuracies):
    best_alpha, best_avg_norm_acc = 0, 0
    progress_file = os.path.join(args.results_dir, "progress.json")

    for alpha in np.arange(0.0, 1.05, 0.05):
        avg_norm_acc = evaluate_alpha(args, encoder, task_vectors, datasets, alpha, best_accuracies)
        if avg_norm_acc > best_avg_norm_acc:
            best_avg_norm_acc, best_alpha = avg_norm_acc, alpha

    with open(progress_file, "w") as f:
        json.dump({"best_alpha": best_alpha, "best_avg_norm_acc": best_avg_norm_acc}, f)

    return best_alpha

def evaluate_scaled_model(args, encoder, task_vectors, datasets, alpha):
    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    blended_encoder = combined_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

    test_accuracies, train_accuracies, fim_log_traces = [], [], []
    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(dataset_name, preprocess, dataset_path, args.batch_size)
        train_loader = dataset.train_loader
        test_loader = dataset.test_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(blended_encoder, head).cuda()

        test_acc = evaluate_model(model, test_loader)
        train_acc = evaluate_model(model, train_loader)

        criterion = torch.nn.CrossEntropyLoss()
        fim_log_trace = compute_fim_log_trace(model, train_loader, criterion, device=args.device)

        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        fim_log_traces.append(fim_log_trace)

    avg_absolute_acc = np.mean(test_accuracies)
    avg_normalized_acc = np.mean([test_acc / best_acc if best_acc != 0 else 0 for test_acc, best_acc in zip(test_accuracies, best_accuracies)])

    return {
        "alpha": alpha,
        "test_accuracies": test_accuracies,
        "train_accuracies": train_accuracies,
        "fim_log_traces": fim_log_traces,
        "avg_absolute_acc": avg_absolute_acc,
        "avg_normalized_acc": avg_normalized_acc
    }

def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_baseline"
    args.data_location = "/kaggle/working/datasets"
    args.results_dir = "/kaggle/working/results"
    args.save = "/kaggle/working/checkpoints_baseline"
    args.batch_size = 32

    datasets
