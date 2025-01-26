import os
import json
import torch
import numpy as np
import warnings
from tqdm import tqdm
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms
from task_vectors import NonLinearTaskVector

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def compute_fim_log_trace(model, dataloader, criterion, device):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    total_samples = 0
    for batch in tqdm(dataloader, desc="Computing FIM"):
        if total_samples >= 2000:
            break

        inputs, labels = batch[0].to(device), batch[1].to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim[name] += param.grad.pow(2)

        total_samples += inputs.size(0)

    fim_trace = 0.0
    for name, fim_value in fim.items():
        fim_trace += fim_value.sum().item()

    fim_log_trace = torch.log(torch.tensor(fim_trace / total_samples))
    return fim_log_trace.item()

def evaluate_on_train(args, encoder, task_vectors, datasets, alpha):
    print(f"\n🔍 Evaluating on Train Datasets with α = {alpha:.2f}")

    combined_vector = task_vectors[0] * alpha
    for vec in task_vectors[1:]:
        combined_vector += vec * alpha

    blended_encoder = combined_vector.apply_to(os.path.join(args.checkpoints_path, "pretrained.pt"))

    train_accuracies = []
    fim_log_traces = []

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

        head = get_classification_head(args, dataset_name).cuda()
        model = ImageClassifier(blended_encoder, head).cuda()

        # Compute Train Accuracy
        acc = evaluate_model(model, train_loader)
        train_accuracies.append(acc)
        print(f"✅ Train Accuracy for {dataset_name}: {acc:.4f}")

        # Compute FIM Log Trace
        criterion = torch.nn.CrossEntropyLoss()
        fim_log_trace = compute_fim_log_trace(model, train_loader, criterion, device=args.device)
        fim_log_traces.append(fim_log_trace)
        print(f"✅ Log Tr[FIM] for {dataset_name}: {fim_log_trace:.4f}")

    return train_accuracies, fim_log_traces

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

    task_vectors = [load_task_vector(args, dataset) for dataset in datasets]

    best_accuracies = [json.load(open(os.path.join(args.results_dir, f"{ds}_results.json")))['validation_accuracy'] for ds in datasets]

    best_alpha_path = os.path.join(args.results_dir, "alpha_results.json")
    if not os.path.exists(best_alpha_path):
        raise FileNotFoundError(f"Best alpha results not found at {best_alpha_path}")

    with open(best_alpha_path, "r") as f:
        alpha_results = json.load(f)
        best_alpha = alpha_results["best_alpha"]

    print(f"🏆 Using Best Alpha (α★): {best_alpha:.2f}")

    train_accuracies, fim_log_traces = evaluate_on_train(args, encoder, task_vectors, datasets, best_alpha)

    results = {
        "train_accuracies": train_accuracies,
        "fim_log_traces": fim_log_traces
    }

    save_path = os.path.join(args.results_dir, "train_results_after_scaling.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Train results after scaling saved to {save_path}")

if __name__ == "__main__":
    main()
