import os
import torch
import json
import argparse
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from datasets.registry import get_dataset
from utils import train_diag_fim_logtr as compute_fim_log_trace
from torchvision import transforms
from task_vectors import NonLinearTaskVector
from args import parse_arguments


def evaluate_model(model, dataloader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            else:
                inputs, labels = batch[0].cuda(), batch[1].cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--pretrained_path", type=str, default="/kaggle/working/checkpoints_batchsize/pretrained.pt")
    parser.add_argument("--finetuned_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="/kaggle/working/results_after_scaling")
    parser.add_argument("--data_location", type=str, default="/kaggle/working/datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Add missing attributes to match other scripts
    args.model = "ViT-B-32__pretrained__openai"
    args.save = args.results_dir
    args.device = "cuda"
    args.cache_dir = None
    args.openclip_cachedir = None

    os.makedirs(args.results_dir, exist_ok=True)

    # Load task vector and apply scaling
    task_vector = NonLinearTaskVector(args.pretrained_path, args.finetuned_path)
    encoder = task_vector.apply_to(args.pretrained_path, scaling_coef=args.alpha).cuda()

    # Load classification head
    head = get_classification_head(args, f"{args.dataset}Val").cuda()
    model = ImageClassifier(encoder, head).cuda()

    # Data transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3) if args.dataset.lower() == "mnist" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data
    dataset = get_dataset(f"{args.dataset}Val", preprocess, args.data_location, batch_size=args.batch_size)
    train_loader = dataset.train_loader
    test_dataset = get_dataset(args.dataset, preprocess, args.data_location, batch_size=args.batch_size)
    test_loader = test_dataset.test_loader

    # Evaluate
    train_acc = evaluate_model(model, train_loader)
    test_acc = evaluate_model(model, test_loader)
    fim_log_trace = compute_fim_log_trace(model, train_loader, torch.nn.CrossEntropyLoss(), device='cuda')

    # Save results
    result = {
        "dataset": args.dataset,
        "alpha": args.alpha,
        "scaled_train_accuracy": train_acc,
        "scaled_test_accuracy": test_acc,
        "fim_log_trace": fim_log_trace
    }

    result_path = os.path.join(args.results_dir, f"{args.dataset}_scaled_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"✅ Results saved to {result_path}")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
