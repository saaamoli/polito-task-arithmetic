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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def save_pretrained_model(args):
    save_path = os.path.join(args.checkpoints_path, "pretrained.pt")
    if os.path.exists(save_path):
        print(f"✅ Pre-trained model already exists at {save_path}. Skipping...")
        return False  # Indicates no need to re-save
    print("🔄 Saving the pre-trained model...")
    encoder = ImageEncoder(args).cuda()
    encoder.save(save_path)
    print(f"✅ Pre-trained model saved at {save_path}")
    return True  # Indicates model was saved

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

def compute_log_trace(model, dataloader, criterion):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    total_samples = 0
    for batch in dataloader:
        if total_samples >= 2000:
            break

        inputs, labels = batch[0].cuda(), batch[1].cuda()
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim[name] += param.grad.pow(2)

        total_samples += inputs.size(0)

    fim_trace = sum(fim[name].sum().item() for name in fim)
    fim_log_trace = torch.log(torch.tensor(fim_trace / total_samples))
    return fim_log_trace.item()

def main():
    args = parse_arguments()
    args.checkpoints_path = "/kaggle/working/checkpoints_updated"
    args.data_location = "/kaggle/working/datasets"
    args.results_dir = "/kaggle/working/results"
    args.save = "/kaggle/working/checkpoints_updated"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    save_pretrained_model(args)
    encoder = ImageEncoder(args).cuda()

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(args, dataset_name)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = get_dataset(f"{dataset_name}Train", preprocess, dataset_path, args.batch_size)
        train_loader = dataset.train_loader

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(encoder, head).cuda()

        train_acc = evaluate_model(model, train_loader)
        print(f"✅ Train Accuracy for {dataset_name}: {train_acc:.4f}")

        criterion = torch.nn.CrossEntropyLoss()
        fim_log_trace = compute_log_trace(model, train_loader, criterion)
        print(f"📊 log Tr[FIM] for {dataset_name}: {fim_log_trace:.4f}")

if __name__ == "__main__":
    main()
