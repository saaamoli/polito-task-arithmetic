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

# Load hyperparameters from hyperparams.json
hyperparams_path = '/kaggle/working/polito-task-arithmetic/hyperparams.json'

if not os.path.exists(hyperparams_path):
    raise FileNotFoundError(f"Hyperparameter configuration file not found at {hyperparams_path}")

with open(hyperparams_path, "r") as f:
    baseline_hyperparams = json.load(f)


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
    max_samples = 5000  # Increased sample size for better approximation
    scaling_factor = 1e3  # Gradient scaling to stabilize small values
    dataloader_iterator = iter(dataloader)

    print(f"Starting FIM computation with dataset size: {len(dataloader.dataset)}")
    while total_samples < max_samples:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)  # Restart iterator if exhausted
            batch = next(dataloader_iterator)

        # Handle batch types
        if isinstance(batch, dict):
            inputs, labels = batch['images'].to(device), batch['labels'].to(device)
        elif isinstance(batch, (list, tuple)):
            inputs, labels = batch[0].to(device), batch[1].to(device)
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        model.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if not loss.requires_grad:
            raise ValueError("Loss does not require gradients. Check the computation graph.")

        loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim[name] += scaling_factor * param.grad.pow(2)  # Scale gradients to stabilize FIM computation

        total_samples += inputs.size(0)
        if total_samples >= max_samples:
            print(f"Processed {total_samples} samples for FIM computation.")
            break

    fim_trace = sum(fim_value.sum().item() for fim_value in fim.values())
    print(f"Raw FIM Trace Sum: {fim_trace}")

    # Normalize and safeguard trace computation
    normalized_trace = max(fim_trace / (total_samples or 1), 1e-6)  # Avoid division by zero
    fim_log_trace = torch.log(torch.tensor(normalized_trace))

    print(f"Normalized FIM Trace: {normalized_trace}, Log Trace: {fim_log_trace.item()}")
    return fim_log_trace.item()

def compute_average_normalized_accuracy(val_accuracies, single_task_accuracies):
    return np.mean([val / single if single != 0 else 0 for val, single in zip(val_accuracies, single_task_accuracies)])

def evaluate_alpha(args, encoder, task_vectors, datasets, alpha, single_task_accuracies):
    print(f"\n🔍 Evaluating alpha = {alpha:.2f}")

    # Combine task vectors with the current alpha
    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    # Apply the combined task vector to the pre-trained encoder
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
        print(f"📊 Validation Accuracy for {dataset_name}: {acc:.4f}")

    avg_norm_acc = compute_average_normalized_accuracy(val_accuracies, single_task_accuracies)
    print(f"📈 Average Normalized Accuracy at alpha {alpha:.2f}: {avg_norm_acc:.4f}")
    return avg_norm_acc, val_accuracies

def evaluate_multitask_metrics(args, encoder, task_vectors, datasets, alpha, train_accuracies, single_task_accuracies):
    print(f"\n🧪 Evaluating Multi-task Metrics with α⋆ = {alpha:.2f}")

    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    blended_encoder = combined_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

    absolute_train_acc = []
    absolute_test_acc = []
    normalized_train_acc = []
    normalized_test_acc = []
    fim_log_traces = []

    for i, dataset_name in enumerate(datasets):
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

        # Compute train/test accuracies
        train_acc = evaluate_model(model, train_loader)
        test_acc = evaluate_model(model, test_loader)
        absolute_train_acc.append(train_acc)
        absolute_test_acc.append(test_acc)

        # Compute normalized accuracies
        normalized_train_acc.append(train_acc / train_accuracies[i])
        normalized_test_acc.append(test_acc / single_task_accuracies[i])

        # Compute FIM log-trace
        criterion = torch.nn.CrossEntropyLoss()
        fim_log_trace = compute_fim_log_trace(model, train_loader, criterion, device=args.device)
        fim_log_traces.append(fim_log_trace)

        print(f"✅ Dataset: {dataset_name} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"📊 Normalized Train Acc: {normalized_train_acc[-1]:.4f} | Normalized Test Acc: {normalized_test_acc[-1]:.4f}")
        print(f"📊 FIM Log Trace: {fim_log_trace:.4f}")

    return absolute_train_acc, absolute_test_acc, normalized_train_acc, normalized_test_acc, fim_log_traces

def main():
    # Parse arguments
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_batchsize"
    args.data_location = "/kaggle/working/datasets"
    args.results_dir = "/kaggle/working/results_batchsize"
    args.save = "/kaggle/working/checkpoints_batchsize"

    # Define the list of datasets
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Save the pre-trained model if it doesn't already exist
    save_pretrained_model(args)

    # Load the pre-trained encoder and task vectors for each dataset
    encoder = ImageEncoder(args).cuda()
    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    # Load single-task metrics
    single_task_accuracies = [
        json.load(open(os.path.join(args.results_dir, f"{dataset}_results.json")))['test_accuracy']
        for dataset in datasets
    ]

    train_accuracies = [
        json.load(open(os.path.join(args.results_dir, f"{dataset}_results.json")))['train_accuracy']
        for dataset in datasets
    ]

    fim_traces = [
        json.load(open(os.path.join(args.results_dir, f"{dataset}_results.json")))['fim_log_trace']
        for dataset in datasets
    ]

    # Search for the best alpha or load it from progress
    progress_file = os.path.join(args.results_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
            best_alpha = progress.get("best_alpha", 0)
            print(f"🔄 Resuming from α = {best_alpha:.2f}")
    else:
        best_alpha, best_avg_norm_acc = 0, 0
        for alpha in np.arange(0.0, 1.05, 0.05):
            # Dynamically set batch size for the first dataset (e.g., "DTD")
            args.batch_size = baseline_hyperparams[datasets[0]]["batch_size"]
            avg_norm_acc, val_accuracies = evaluate_alpha(
                args, encoder, task_vectors, datasets, alpha, single_task_accuracies
            )
            if avg_norm_acc > best_avg_norm_acc:
                best_avg_norm_acc, best_alpha = avg_norm_acc, alpha

        # Save progress
        with open(progress_file, "w") as f:
            json.dump({"best_alpha": best_alpha, "best_avg_norm_acc": best_avg_norm_acc}, f)

    print(f"🏆 Best Alpha (α★): {best_alpha:.2f}")

    # Evaluate multi-task metrics after task addition
    absolute_train_acc, absolute_test_acc, normalized_train_acc, normalized_test_acc, fim_log_traces_scaled = (
        evaluate_multitask_metrics(args, encoder, task_vectors, datasets, best_alpha, train_accuracies, single_task_accuracies)
    )

    # Save the results
    results = {
        "alpha": best_alpha,
        "absolute_train_accuracy": absolute_train_acc,
        "absolute_test_accuracy": absolute_test_acc,
        "normalized_train_accuracy": normalized_train_acc,
        "normalized_test_accuracy": normalized_test_acc,
        "fim_log_traces_scaled": fim_log_traces_scaled
    }
    save_path = os.path.join(args.results_dir, "task_addition_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Task addition metrics saved to {save_path}")



if __name__ == "__main__":
    main()
