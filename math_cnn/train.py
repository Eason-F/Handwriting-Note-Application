import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import (
    HASYDataset,
    fold_paths,
    load_symbols,
)
from .model import SymbolCNN


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


# Arguments

def parse_args():
    parser = argparse.ArgumentParser(description="Train the writing CNN")

    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/math_cnn.pt"))
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    
    persistent = args.workers > 0

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "persistent_workers": persistent,
    }

    if persistent:
        loader_args["prefetch_factor"] = 2

    symbols = load_symbols(args.data)
    train_csv, test_csv = fold_paths(args.data, args.fold)
    hasy_train = HASYDataset(train_csv, symbols, augment=False)
    hasy_test = HASYDataset(test_csv, symbols, augment=False)
    
    test_loader = DataLoader(
        hasy_test,
        shuffle=False,
        **loader_args
    )
    train_loader = DataLoader(
        hasy_train,
        shuffle=True,
        **loader_args
    )
    
    sample_counts = HASYDataset.get_total_class_samples(symbols)
    total_samples = np.sum(sample_counts)
    weights = total_samples / (sample_counts + 10.0)
    weights = (weights / np.sum(weights)) * len(hasy_train)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = SymbolCNN(len(symbols)).to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=weights_tensor
    )
    optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.patience,
        min_lr=1e-6,
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes: {len(symbols)}")
    print(f"HASY training samples: {len(hasy_train):,}")
    print(f"HASY test samples: {len(hasy_test):,}")

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()

        loss_total = correct = samples = 0

        for batch_number, (images, targets) in enumerate(train_loader, start=1):
            batch_started = time.perf_counter()

            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            scores = model(images)
            loss = loss_function(scores, targets)

            loss.backward()
            optimizer.step()

            predictions = scores.argmax(dim=1)
            batch_size = targets.size(0)

            loss_total += loss.item() * batch_size
            correct += (predictions == targets).sum().item()
            samples += batch_size

            if batch_number % args.log_every == 0 or batch_number == len(train_loader):
                elapsed = time.perf_counter() - batch_started
                average_loss = loss_total / samples
                accuracy = correct / samples
                progress = batch_number / len(train_loader)

                print(
                    f"  Batch {batch_number:>{len(str(len(train_loader)))}}/"
                    f"{len(train_loader)} "
                    f"({progress:6.1%}) "
                    f"loss={average_loss:.4f} "
                    f"acc={accuracy:.2%} "
                    f"time={elapsed:.2f}s",
                    flush=True,
                )

        train_loss = loss_total / samples
        train_accuracy = correct / samples

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_function,
            device,
        )

        scheduler.step(test_accuracy)

        elapsed = time.perf_counter() - started
        samples_per_second = samples / elapsed
        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"train_accuracy (augmented)={train_accuracy:.2%} "
            f"test_accuracy={test_accuracy:.2%} "
            f"lr={learning_rate:.2e} "
            f"time={elapsed:.1f}s "
            f"samples/s={samples_per_second:.0f}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "number_of_classes": len(symbols),
                    "symbols": [
                        {"symbol_id": symbol.symbol_id, "latex": symbol.latex}
                        for symbol in symbols
                    ],
                    "fold": args.fold,
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                    "train_accuracy": test_accuracy,
                },
                args.output,
            )
            print(f"  Saved improved checkpoint to {args.output}")

    print(f"Best test accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
