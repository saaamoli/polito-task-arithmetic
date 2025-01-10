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
