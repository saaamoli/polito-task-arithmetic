import os
import time
import torch
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms  # Import transforms for preprocessing

def fine_tune_on_dataset(args, dataset_name, num_epochs):
    print(f"Fine-tuning on {dataset_name} for {num_epochs} epochs...")

    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),         # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load dataset with transforms
    dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=preprocess,  # Pass the transform here
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )
    train_loader = get_dataloader(dataset, is_train=True, args=args)

    # Debugging: Print dataset details
    if hasattr(dataset, "train_dataset"):
        print(f"Dataset loaded with {len(dataset.train_dataset)} samples.")
    elif hasattr(dataset, "__len__"):
        print(f"Dataset loaded with {len(dataset)} samples.")
    else:
        print("Unable to determine dataset size. Proceeding...")

    print(f"Batch size: {args.batch_size}")

    # Load pre-trained model
    encoder = ImageEncoder(args)  # Pre-trained model
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    model.freeze_head()

    # Define optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    # Fine-tuning loop
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"], data["labels"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader):.4f}")

    # Save the fine-tuned encoder
    os.makedirs(args.save, exist_ok=True)
    save_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    model.image_encoder.save(save_path)
    print(f"Fine-tuned model saved to {save_path}")

if __name__ == "__main__":
    args = parse_arguments()
    dataset_epochs = {
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SVHN": 4
    }

    # Iterate over all datasets
    for dataset_name, num_epochs in dataset_epochs.items():
        args.train_dataset = [dataset_name]  # Set the current dataset
        fine_tune_on_dataset(args, dataset_name, num_epochs)
