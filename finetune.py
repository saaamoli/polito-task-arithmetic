import os
import torch
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms  # Import transforms for preprocessing

def fine_tune(args):
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),         # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load dataset with transforms
    dataset = get_dataset(
        f"{args.train_dataset[0]}Val",
        preprocess=preprocess,  # Pass the transform here
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )
    train_loader = get_dataloader(dataset, is_train=True, args=args)

    # Load pre-trained model
    encoder = ImageEncoder(args)  # Pre-trained model
    head = get_classification_head(args, f"{args.train_dataset[0]}Val")
    model = ImageClassifier(encoder, head)
    model.freeze_head()

    # Define optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

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
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(os.path.join(args.save, f"{args.train_dataset[0]}_finetuned.pt"))
    print(f"Fine-tuned model saved to {os.path.join(args.save, f'{args.train_dataset[0]}_finetuned.pt')}")

if __name__ == "__main__":
    args = parse_arguments()
    fine_tune(args)
