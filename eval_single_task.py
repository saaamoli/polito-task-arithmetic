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
