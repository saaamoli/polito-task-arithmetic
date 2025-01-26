import os
import torch
import json
import numpy as np
from tqdm import tqdm
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms

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

def save_task_vector(args, dataset_name):
    """Save the task vector for a dataset."""
    pretrained_checkpoint = os.path.join(args.checkpoints_path, "pretrained.pt")
    finetuned_checkpoint = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    task_vector_path = os.path.join(args.checkpoints_path, f"{dataset_name}_task_vector.pt")

    if not os.path.exists(pretrained_checkpoint):
        raise FileNotFoundError(f"Pre-trained checkpoint not found at {pretrained_checkpoint}")
    if not os.path.exists(finetuned_checkpoint):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found at {finetuned_checkpoint}")

    if os.path.exists(task_vector_path):
        print(f"✅ Task vector for {dataset_name} already exists at {task_vector_path}. Skipping...")
        return

    print(f"🔄 Generating task vector for {dataset_name}...")
    pretrained_model = torch.load(pretrained_checkpoint)
    finetuned_model = torch.load(finetuned_checkpoint)

    # Compute the task vector as the difference between fine-tuned and pre-trained weights
    task_vector = {name: finetuned_model[name] - pretrained_model[name]
                   for name in pretrained_model if name in finetuned_model}

    torch.save(task_vector, task_vector_path)
    print(f"✅ Task vector saved at {task_vector_path}")


def load_task_vector(args, dataset_name):
    """Load the task vector for a dataset."""
    task_vector_path = os.path.join(args.checkpoints_path, f"{dataset_name}_task_vector.pt")

    if not os.path.exists(task_vector_path):
        save_task_vector(args, dataset_name)

    print(f"🔄 Loading task vector for {dataset_name}")
    return torch.load(task_vector_path)

def compute_train_accuracy_and_fim(model, train_loader, device="cuda"):
    # Compute train accuracy
    correct, total = 0, 0
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_samples = 0

    for batch in tqdm(train_loader, desc="Processing Train Batch"):
        inputs, labels = batch[0].to(device), batch[1].to(device)

        model.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Compute FIM
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim[name] += param.grad.pow(2)

        total_samples += inputs.size(0)

    train_accuracy = correct / total
    fim_trace = sum(fim[name].sum().item() for name in fim)
    fim_log_trace = np.log(fim_trace / total_samples)
    return train_accuracy, fim_log_trace

def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_updated"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    alpha = 0.3  # Use the best alpha from your eval_task_addition.py results

    results = {}

    for dataset_name in datasets:
        print(f"\n--- Processing {dataset_name} ---")
        dataset_path = resolve_dataset_path(args, dataset_name)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
        train_loader = dataset.train_loader

        model = load_task_vector_model(args, dataset_name, alpha)
        train_acc, fim_log_trace = compute_train_accuracy_and_fim(model, train_loader)

        results[dataset_name] = {
            "train_accuracy": train_acc,
            "fim_log_trace_train": fim_log_trace
        }

    save_path = "/kaggle/working/scaled_train_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Results saved to {save_path}")

if __name__ == "__main__":
    main()
