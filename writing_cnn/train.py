import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from .data import (
    EMNISTDataset,
    EMNIST_BYCLASS_CHARACTERS,
    fold_paths,
    load_emnist_source,
)
from .model import WritingCNN


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, loss_function, device):
    model.eval()
    loss_total = 0.0
    correct = 0
    sample_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            scores = model(images)
            loss_total += loss_function(scores, targets).item() * targets.size(0)
            correct += (scores.argmax(dim=1) == targets).sum().item()
            sample_count += targets.size(0)
    return loss_total / sample_count, correct / sample_count


def parse_args():
    parser = argparse.ArgumentParser(description="Train the writing CNN")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/writing_cnn.pt"))
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    parser.add_argument(
        "--tfds-data", type=Path, default=Path("data/tfds/parquet")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(7)
    torch.manual_seed(7)
    device = choose_device(args.device)

    symbols = list(EMNIST_BYCLASS_CHARACTERS)
    train_csv, test_csv = fold_paths(args.data, args.fold)
    
    print("Loading EMNIST ByClass through TFDS (first run downloads it)...")
    emnist_train = EMNISTDataset(
        load_emnist_source(args.tfds_data, "train"),
    )
    emnist_test = EMNISTDataset(
        load_emnist_source(args.tfds_data, "test"),
    )

    sample_weights = torch.full((len(emnist_train),), 0.5 / len(emnist_train))
    
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=2 * len(emnist_train),
        replacement=True,
    )
    emnist_train_loader = DataLoader(
        emnist_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
    )
    emnist_test_loader = DataLoader(
        emnist_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    
    
    all_labels = np.array([int(label) for image, label in emnist_train], dtype=np.int64)
    classes, counts = np.unique(all_labels, return_counts=True)
    num_classes = len(classes)

    total_samples = len(all_labels)
    weights = total_samples / (num_classes * counts)
    class_weights_tensor = torch.FloatTensor(weights).to(device)

    model = WritingCNN(len(symbols)).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes: {len(symbols)}")
    print(f"EMNIST training samples: {len(emnist_train):,}")
    print(f"EMNIST test samples: {len(emnist_test):,}")
    print(f"Balanced samples drawn per epoch: {len(sampler):,}")

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch_number, (images, targets) in enumerate(emnist_train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            
            scores = model(images)
            loss = loss_function(scores, targets)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            sample_count += targets.size(0)
            if batch_number % 100 == 0:
                print(
                    f"  epoch {epoch}: batch {batch_number}/{len(emnist_train_loader)} "
                    f"loss={running_loss / sample_count:.4f}",
                    flush=True,
                )

        emnist_loss, emnist_accuracy = evaluate(
            model, emnist_test_loader, loss_function, device
        )
        test_loss = emnist_loss
        test_accuracy = emnist_accuracy
        train_loss = running_loss / sample_count
        seconds = time.perf_counter() - started
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            + (
                f"emnist_accuracy={emnist_accuracy:.2%} "
                if emnist_accuracy is not None
                else ""
            )
            + f"selection_accuracy={test_accuracy:.2%} "
            f"time={seconds:.1f}s"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "number_of_classes": len(symbols),
                    "fold": args.fold,
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                    "emnist_accuracy": emnist_accuracy,
                },
                args.output,
            )
            print(f"  Saved improved checkpoint to {args.output}")

    print(f"Best test accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
