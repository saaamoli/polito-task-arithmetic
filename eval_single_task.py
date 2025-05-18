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

# Load hyperparameters
script_dir = os.path.abspath(os.path.dirname(__file__))
hyperparams_path = os.path.join(script_dir, "hyperparams.json")
with open(hyperparams_path, "r") as f:
    baseline_hyperparams = json.load(f)

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

def get_encoder_path(args, dataset_name):
    filename_map = {
        "val": f"{dataset_name}_bestvalacc.pt",
        "fim": f"{dataset_name}_bestfim.pt",
        "last": f"{dataset_name}_finetuned.pt"
    }
    filename = filename_map[args.selection_mode]
    return os.path.join(args.checkpoints_path, filename)

def load_finetuned_model(args, dataset_name):
    encoder_path = get_encoder_path(args, dataset_name)
    head_path = os.path.join(args.checkpoints_path, f"head_{dataset_name}Val.pt")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"❌ Encoder checkpoint not found at {encoder_path}")
    encoder = torch.load(encoder_path, map_location="cuda")

    if os.path.exists(head_path):
        print(f"✅ Loading existing classification head for {dataset_name} from {head_path}")
        head = torch.load(head_path, map_location="cuda").cuda()
    else:
        print(f"⚠️ Classification head not found. Regenerating for {dataset_name}...")
        head = get_classification_head(args, dataset_name).cuda()
        os.makedirs(os.path.dirname(head_path), exist_ok=True)
        head.save(head_path)

    return ImageClassifier(encoder, head).cuda()

def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            inputs, labels = batch["images"].cuda(), batch["labels"].cuda()
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
    save_path = os.path.join(args.results_dir, f"{dataset_name}_results_{args.selection_mode}.json")
    if os.path.exists(save_path):
        print(f"✅ Results already exist at {save_path}. Skipping...")
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

    fim_trace = train_diag_fim_logtr(args, model, dataset_name)
    print(f"📊 logTr[FIM] for {dataset_name}: {fim_trace:.4f}")

    results = {
        "dataset": dataset_name,
        "selection_mode": args.selection_mode,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "fim_log_trace": fim_trace
    }

    save_results(results, save_path)
    args.data_location = original_data_location

def main():
    args = parse_arguments()

    project_root = os.path.abspath(args.data_location)
    args.data_location = os.path.join(project_root, "datasets")

    if args.save is None:
        args.save = os.path.join(project_root, f"checkpoints_{args.exp_name or 'default'}")

    args.checkpoints_path = args.save
    args.results_dir = args.save.replace("checkpoints", "results")
    os.makedirs(args.results_dir, exist_ok=True)

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    for dataset_name in datasets:
        print(f"\n--- Evaluating {dataset_name} with {args.selection_mode} checkpoint ---")
        args.batch_size = baseline_hyperparams[dataset_name]["batch_size"]
        evaluate_and_save(args, dataset_name)

if __name__ == "__main__":
    main()
