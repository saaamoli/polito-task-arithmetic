import os
import torch
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments

def load_finetuned_model(args, dataset_name):
    # Path to the fine-tuned encoder checkpoint
    encoder_checkpoint_path = os.path.join(args.checkpoints_path, f"{dataset_name}_finetuned.pt")
    if not os.path.exists(encoder_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_checkpoint_path}")
    
    # Load the fine-tuned encoder
    encoder = ImageEncoder(args).cuda()
    encoder.load_state_dict(torch.load(encoder_checkpoint_path))
    
    # Load the classification head for the dataset
    head = get_classification_head(args, dataset_name).cuda()
    
    # Combine encoder and head into a classifier
    model = ImageClassifier(encoder, head).cuda()
    
    return model
from datasets.registry import get_dataset
from torchvision import transforms

def load_datasets(args, dataset_name):
    # Define standard preprocessing transforms (similar to fine-tuning)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the validation dataset
    val_dataset = get_dataset(
        f"{dataset_name}Val",  # Validation split
        preprocess=preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )

    # Load the test dataset
    test_dataset = get_dataset(
        dataset_name,  # Test split
        preprocess=preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )

    return val_dataset, test_dataset

import torch
from tqdm import tqdm

def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation during evaluation
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = batch["images"].cuda(), batch["labels"].cuda()
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class

            # Count correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total  # Calculate accuracy
    return accuracy

def run_evaluation(args, dataset_name, model):
    # Load validation and test datasets
    val_dataset, test_dataset = load_datasets(args, dataset_name)

    # Evaluate on Validation Set
    val_accuracy = evaluate_model(model, val_dataset.test_loader)
    print(f"Validation Accuracy on {dataset_name}: {val_accuracy * 100:.2f}%")

    # Evaluate on Test Set
    test_accuracy = evaluate_model(model, test_dataset.test_loader)
    print(f"Test Accuracy on {dataset_name}: {test_accuracy * 100:.2f}%")

    return val_accuracy, test_accuracy

import json
import os

def save_results(dataset_name, val_accuracy, test_accuracy, save_dir="/kaggle/working/results"):
    # Ensure the results directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare the results dictionary
    results = {
        "dataset": dataset_name,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy
    }
    
    # Define the file path
    results_file = os.path.join(save_dir, f"{dataset_name}_results.json")
    
    # Save results to JSON
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")


def evaluate_and_save(args, dataset_name, model):
    # Run evaluation
    val_accuracy, test_accuracy = run_evaluation(args, dataset_name, model)
    
    # Save the results
    save_results(dataset_name, val_accuracy, test_accuracy)



def evaluate_all_datasets(args):
    # List of all datasets we fine-tuned on
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    for dataset_name in datasets:
        print(f"\n--- Evaluating on {dataset_name} ---")
        
        # Load the fine-tuned model for the dataset
        model = load_finetuned_model(args, dataset_name)
        
        # Evaluate and save results
        evaluate_and_save(args, dataset_name, model)



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run evaluation for all datasets
    evaluate_all_datasets(args)


