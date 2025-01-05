import os
import time
import torch
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms

def fine_tune_on_dataset(args, dataset_name, num_epochs):
    print(f"\n==== Fine-tuning on {dataset_name} for {num_epochs} epochs ====\n")
    print(f"Dataset location: {args.data_location}")
    print(f"Expected path: {os.path.join(args.data_location, dataset_name.lower())}")


    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Dataset location passed to DTD: {os.path.join(args.data_location, dataset_name.lower())}")

    # Load dataset with transforms
    dataset = get_dataset(
    dataset_name,
    preprocess=preprocess,
    location=os.path.join(args.data_location, dataset_name.lower()),
    batch_size=args.batch_size,
    num_workers=2
    )

    train_loader = get_dataloader(dataset, is_train=True, args=args)

    # Debugging: Print dataset details
    dataset_size = len(dataset.train_dataset) if hasattr(dataset, "train_dataset") else len(dataset)
    print(f"Dataset loaded with {dataset_size} samples.")
    print(f"Batch size: {args.batch_size}")

    # Load pre-trained model
    encoder = ImageEncoder(args).cuda()
    head = get_classification_head(args, f"{dataset_name}Val").cuda()
    model = ImageClassifier(encoder, head).cuda()
    model.freeze_head()

    # Define optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    # Fine-tuning loop
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}...")
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].cuda(), data["labels"].cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}, Time = {time.time() - batch_start:.2f}s", end="\r")

        print(f"\nEpoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader):.4f}")

    # Save the fine-tuned encoder
    save_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(save_path)
    print(f"Fine-tuned model saved to {save_path}")
    print(f"Time taken for {dataset_name}: {time.time() - start_time:.2f}s\n")

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

    # Fine-tune on each dataset
    for dataset_name, num_epochs in dataset_epochs.items():
        args.train_dataset = [dataset_name]
        fine_tune_on_dataset(args, dataset_name, num_epochs)
