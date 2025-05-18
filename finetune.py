import os
import json
import torch
from tqdm import tqdm
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from torchvision import transforms
from utils import train_diag_fim_logtr

def resolve_dataset_path(args, dataset_name):
    base_path = args.data_location
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "dtd":
        return os.path.join(base_path)  
    elif dataset_name_lower == "eurosat":
        return base_path
    elif dataset_name_lower == "mnist":
        return os.path.join(base_path, "MNIST", "raw")
    elif dataset_name_lower == "gtsrb":
        return os.path.join(base_path, "gtsrb")
    elif dataset_name_lower == "resisc45":
        return base_path
    elif dataset_name_lower == "svhn":
        return os.path.join(base_path, "svhn")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            x, y = batch["images"].cuda(), batch["labels"].cuda()
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def fine_tune_on_dataset(args, dataset_name, num_epochs):
    print(f"\n🔧 Fine-tuning on {dataset_name}")
    path = resolve_dataset_path(args, dataset_name)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3) if dataset_name.lower() == "mnist" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = get_dataset(f"{dataset_name}Val", preprocess=preprocess, location=path, batch_size=args.batch_size, num_workers=2)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)


    encoder = ImageEncoder(args).cuda()
    head = get_classification_head(args, dataset_name).cuda()
    model = ImageClassifier(encoder, head).cuda()
    model.train()

    optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.ls)

    best_val_acc = 0.0
    best_fim_score = -float("inf")

    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            batch = maybe_dictionarize(batch)
            x, y = batch["images"].cuda(), batch["labels"].cuda()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_accuracy = evaluate_model(model, val_loader)
        print(f"✅ Val accuracy: {val_accuracy:.4f}")

        # Save best validation accuracy model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_bestvalacc.pt"))
            print("💾 Saved best validation accuracy checkpoint.")

        # Save best FIM model
        try:
            fim_trace = train_diag_fim_logtr(args, model, dataset_name)
            print(f"📊 logTr[FIM]: {fim_trace:.4f}")
            if fim_trace > best_fim_score:
                best_fim_score = fim_trace
                model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_bestfim.pt"))
                print("💾 Saved best FIM trace checkpoint.")
        except Exception as e:
            print(f"⚠️ Could not compute FIM for {dataset_name} epoch {epoch+1}: {e}")

    # Save final checkpoint (last epoch)
    model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_finetuned.pt"))
    print(f"📦 Saved final model for {dataset_name} at last epoch.")
    print(f"🔚 Best Val Acc: {best_val_acc:.4f} | Best FIM: {best_fim_score:.4f}")

    # Save per-dataset metrics summary
    metrics_summary = {
        "dataset": dataset_name,
        "best_val_acc": best_val_acc,
        "best_fim_log_trace": best_fim_score,
        "checkpoint_val": f"{dataset_name}_bestvalacc.pt",
        "checkpoint_fim": f"{dataset_name}_bestfim.pt",
        "checkpoint_last": f"{dataset_name}_finetuned.pt"
    }

    with open(os.path.join(args.results_dir, f"metrics_{dataset_name}.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"📝 Saved summary metrics to metrics_{dataset_name}.json")

def main():
    args = parse_arguments()
    project_root = os.path.abspath(args.data_location)
    args.data_location = os.path.join(project_root, "datasets")

    if args.save is None:
        if args.exp_name:
            args.save = os.path.join(project_root, f"checkpoints_{args.exp_name}")
        else:
            args.save = os.path.join(project_root, "checkpoints_default")

    args.checkpoints_path = args.save
    args.results_dir = args.save.replace("checkpoints", "results")
    os.makedirs(args.results_dir, exist_ok=True)

    # Load baseline epochs from hyperparams.json
    script_dir = os.path.abspath(os.path.dirname(__file__))
    hyperparams_path = os.path.join(script_dir, "hyperparams.json")
    with open(hyperparams_path, "r") as f:
        baseline_hyperparams = json.load(f)

    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    for dataset_name in datasets:
        num_epochs = baseline_hyperparams[dataset_name]["epochs"]
        args.batch_size = baseline_hyperparams[dataset_name]["batch_size"]
        args.lr = baseline_hyperparams[dataset_name]["learning_rate"]
        args.wd = baseline_hyperparams[dataset_name]["weight_decay"]
        print(f"\n🚀 Starting training for {dataset_name} with {num_epochs} epochs")
        fine_tune_on_dataset(args, dataset_name, num_epochs)

if __name__ == "__main__":
    main()
