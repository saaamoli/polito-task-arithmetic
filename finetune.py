import os
import time
import torch
import sys
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from args import parse_arguments
from utils import train_diag_fim_logtr  # ✅ For logTr[FIM]

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

def fine_tune_on_dataset(args, dataset_name, num_epochs, learning_rate, batch_size, weight_decay, log_path):
    print(f"\n==== Fine-tuning on {dataset_name} with LR={learning_rate}, Batch Size={batch_size}, WD={weight_decay} ====\n")

    # ✅ Temporarily adjust dataset path
    original_data_location = args.data_location
    args.data_location = resolve_dataset_path(args, dataset_name)

    # ✅ Check if all Phase 4 checkpoints exist, skip if so
    checkpoint_finetuned = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
    checkpoint_valacc = os.path.join(args.save, f"{dataset_name}_bestvalacc.pt")
    checkpoint_fim = os.path.join(args.save, f"{dataset_name}_bestfim.pt")
    
   completion_flag = os.path.join(args.results_dir, f"finetune_complete_{dataset_name}.txt")
    if os.path.exists(completion_flag):
        print(f"✅ Skipping {dataset_name} — already marked as complete.")
        args.data_location = original_data_location
        return


    encoder = ImageEncoder(args).to(args.device)

    # ✅ Save pretrained encoder once
    pretrained_path = os.path.join(args.save, "pretrained.pt")
    if not os.path.exists(pretrained_path):
        print(f"✅ Saving pretrained encoder to {pretrained_path}")
        encoder.save(pretrained_path)
    else:
        print(f"ℹ️ Pretrained encoder already exists at {pretrained_path}")

    preprocess = encoder.train_preprocess

    dataset = get_dataset(f"{dataset_name}Val", preprocess=preprocess, location=args.data_location, batch_size=batch_size, num_workers=2)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    head = get_classification_head(args, dataset_name).to(args.device)
    model = ImageClassifier(encoder, head).to(args.device)

    # ✅ Freeze the classification head
    model.freeze_head()

    optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # ✅ Phase 4 trackers
    best_val_acc = 0.0
    best_fim_score = -float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].to(args.device), data["labels"].to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                data = maybe_dictionarize(batch)
                inputs, labels = data["images"].to(args.device), data["labels"].to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        # ✅ Save best validation model
        if args.finetune_mode in ("valacc", "both") and (epoch == 0 or val_accuracy > best_val_acc):
            best_val_acc = val_accuracy
            model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_bestvalacc.pt"))
            print("💾 Saved best validation accuracy checkpoint.")
        
        # ✅ Save best FIM log-trace model
        if args.finetune_mode in ("fim", "both"):
            try:
                fim_trace = train_diag_fim_logtr(args, model, dataset_name)
                print(f"📊 logTr[FIM]: {fim_trace:.4f}")
                if epoch == 0 or fim_trace > best_fim_score:
                    best_fim_score = fim_trace
                    model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_bestfim.pt"))
                    print("💾 Saved best FIM trace checkpoint.")
            except Exception as e:
                print(f"⚠️ Could not compute FIM for {dataset_name} epoch {epoch+1}: {e}")


    # ✅ Save final model
    os.makedirs(args.save, exist_ok=True)
    model.image_encoder.save(os.path.join(args.save, f"{dataset_name}_finetuned.pt"))
    print(f"✅ Fine-tuned model saved to {os.path.join(args.save, f'{dataset_name}_finetuned.pt')}")

    # ✅ Save phase 4 metrics summary
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

    # ✅ Restore original path
    args.data_location = original_data_location
    # Save "I finished this dataset" signal
    with open(os.path.join(args.results_dir, f"finetune_complete_{dataset_name}.txt"), "w") as f:
        f.write("done")

    # Save "I finished this dataset" signal
    with open(os.path.join(args.results_dir, f"finetune_complete_{dataset_name}.txt"), "w") as f:
        f.write("done")



if __name__ == "__main__":
    args = parse_arguments()

    # 🌍 Portable path handling using --data-location as project root
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    hyperparams_path = os.path.join(os.path.dirname(__file__), "hyperparams.json")
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"Hyperparameter configuration file not found at {hyperparams_path}")

    with open(hyperparams_path, "r") as f:
        baseline_hyperparams = json.load(f)

    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    log_path = os.path.join(args.results_dir, "weight_results.json")

    for dataset_name, num_epochs in dataset_epochs.items():
        hyperparams = baseline_hyperparams[dataset_name]
        fine_tune_on_dataset(
            args,
            dataset_name,
            num_epochs,
            hyperparams["learning_rate"],
            hyperparams["batch_size"],
            hyperparams["weight_decay"],
            log_path,
        )
