import os
import sys
import torch

sys.path.append('/kaggle/working/polito-task-arithmetic')

from args import parse_arguments
from modeling import ImageEncoder

def main():
    args = parse_arguments()
    args.model = "ViT-B-32"
    args.device = "cuda"
    args.openclip_cachedir = "/root/.cache/open_clip"
    args.save = "/kaggle/working/checkpoints_baseline"

    pretrained_path = os.path.join(args.save, "pretrained.pt")
    if os.path.exists(pretrained_path):
        print(f"ℹ️ pretrained.pt already exists at: {pretrained_path}")
    else:
        encoder = ImageEncoder(args).to(args.device)
        encoder.save(pretrained_path)
        print(f"✅ Saved full ImageEncoder to: {pretrained_path}")

if __name__ == "__main__":
    main()
