import os
import argparse
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model on a specific dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to fine-tune on")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--epochs", type=int, required=True, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for SGD")
    return parser.parse_args()

def fine_tune(args):
    # Load dataset
    dataset = get_dataset(
        f"{args.dataset}Val",
        preprocess=None,  # Replace with model-specific preprocess if needed
        location=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    train_loader = get_dataloader(dataset, is_train=True, args=args)

    # Load pre-trained model
    encoder = ImageEncoder(args)  # Pre-trained CLIP ViT-B/32
    head = get_classification_head(args, f"{args.dataset}Val")
    model = ImageClassifier(encoder, head)
    model.freeze_head()  # Keep classification head frozen

    # Define optimizer and loss
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    # Fine-tuning loop
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in train_loader:
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"], data["labels"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the fine-tuned encoder
    os.makedirs(args.output_dir, exist_ok=True)
    model.image_encoder.save(os.path.join(args.output_dir, f"{args.dataset}_finetuned.pt"))
    print(f"Fine-tuned model saved to {os.path.join(args.output_dir, f'{args.dataset}_finetuned.pt')}")

if __name__ == "__main__":
    args = parse_arguments()
    fine_tune(args)
