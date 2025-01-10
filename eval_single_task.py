import os
import json
import torch
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments

def load_finetuned_model(args, dataset_name):
    """
    Load the fine-tuned encoder and the classification head for the given dataset.
    """
    # ✅ Path to the fine-tuned encoder checkpoint
    encoder_checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    
    if not os.path.exists(encoder_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_checkpoint_path}")
    
    # ✅ Load the fine-tuned encoder
    encoder = torch.load(encoder_checkpoint_path).cuda()

    # ✅ Correct dataset path for classification head
    dataset_path = resolve_dataset_path(args, dataset_name)
    args.data_location = dataset_path  # Force correct dataset path

    # ✅ Load the classification head for the dataset
    head = get_classification_head(args, dataset_name).cuda()
    
    # ✅ Combine encoder and head into a classifier
    model = ImageClassifier(encoder, head).cuda()
    
    return model



def resolve_dataset_path(args, dataset_name):
    """
    Resolves the correct dataset path for each dataset.
    """
    base_path = args.data_location
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "dtd":
        return os.path.join(base_path, "dtd")
    elif dataset_name_lower == "eurosat":
        return os.path.join(base_path, "EuroSAT_splits")  # ✅ Corrected path
    elif dataset_name_lower == "mnist":
        return os.path.join(base_path, "MNIST", "raw")
    elif dataset_name_lower == "gtsrb":
        return os.path.join(base_path, "gtsrb")
    elif dataset_name_lower == "resisc45":
        return os.path.join(base_path, "resisc45")
    elif dataset_name_lower == "svhn":
        return os.path.join(base_path, "svhn")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def evaluate_model(model, dataloader):
    """
    Evaluates the model on the provided DataLoader and calculates accuracy.
    """
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # ✅ Handle both dict and tuple formats
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

    accuracy = correct / total
    return accuracy



def save_results(results, save_path):
    """
    Saves evaluation results to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


from torchvision import transforms

def evaluate_and_save(args, dataset_name):
    """
    Evaluates the fine-tuned model on validation and test datasets and saves the results.
    """
    dataset_path = resolve_dataset_path(args, dataset_name)

    # ✅ Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),          # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # ✅ Load validation and test datasets with proper transforms
    dataset = get_dataset(f"{dataset_name}Val", preprocess, dataset_path, args.batch_size)
    val_loader = dataset.train_loader
    test_loader = dataset.test_loader

    # ✅ Load the fine-tuned model
    model = load_finetuned_model(args, dataset_name)

    # ✅ Evaluate the model
    val_acc = evaluate_model(model, val_loader)
    test_acc = evaluate_model(model, test_loader)

    # ✅ Prepare results
    results = {
        "dataset": dataset_name,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc
    }

    # ✅ Save results
    save_path = os.path.join(args.results_dir, f"{dataset_name}_results.json")
    save_results(results, save_path)




def main():
    args = parse_arguments()
    
    # ✅ Ensure consistent argument names
    args.checkpoints_path = "/kaggle/working/checkpoints"
    args.results_dir = "/kaggle/working/results"
    args.data_location = "/kaggle/working/datasets"
    args.batch_size = 32

    # ✅ List of datasets to evaluate
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    for dataset_name in datasets:
        print(f"\n--- Evaluating {dataset_name} ---")
        evaluate_and_save(args, dataset_name)


if __name__ == "__main__":
    main()
