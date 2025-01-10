import os
import json
import torch
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments

def load_finetuned_model(args, dataset_name):
    # ✅ Path to the fine-tuned encoder checkpoint
    encoder_checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    
    if not os.path.exists(encoder_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_checkpoint_path}")
    
    # ✅ Load the fine-tuned encoder
    encoder = ImageEncoder(args).cuda()
    encoder.load_state_dict(torch.load(encoder_checkpoint_path))  # <-- Loading checkpoint
    
    # ✅ Load the classification head for the dataset
    head = get_classification_head(args, dataset_name).cuda()
    
    # ✅ Combine encoder and head into a classifier
    model = ImageClassifier(encoder, head).cuda()
    
    return model


def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")

def evaluate_and_save(dataset_name, args):
    dataset = get_dataset(f"{dataset_name}Val", None, args.data_location, args.batch_size)
    val_loader = dataset.train_loader
    test_loader = dataset.test_loader
    
    model = load_finetuned_model(dataset_name, args)
    
    val_acc = evaluate_model(model, val_loader)
    test_acc = evaluate_model(model, test_loader)
    
    results = {
        "dataset": dataset_name,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc
    }
    
    save_path = os.path.join(args.results_dir, f"{dataset_name}_results.json")
    save_results(results, save_path)

def main():
    args = parse_arguments()
    args.checkpoint_dir = "/kaggle/working/checkpoints"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")
        evaluate_and_save(dataset_name, args)

if __name__ == "__main__":
    main()
