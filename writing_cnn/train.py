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
from torchmetrics.classification import MulticlassConfusionMatrix

from .data import EMNISTDataset, EMNIST_BYCLASS_CHARACTERS, fold_paths, load_emnist_source
from .model import WritingCNN


# Device

def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# Evaluation

def evaluate(model, loader, loss_function, device, number_of_classes):
    model.eval()
    loss_total = correct = samples = 0
    confusion = MulticlassConfusionMatrix(num_classes=number_of_classes).to(device)

    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            scores = model(images)
            predictions = scores.argmax(dim=1)

            loss_total += loss_function(scores, targets).item() * targets.size(0)
            correct += (predictions == targets).sum().item()
            samples += targets.size(0)
            confusion.update(predictions, targets)

    return loss_total / samples, correct / samples, confusion.compute().cpu().numpy()


# Reporting

def print_top_confusions(matrix, symbols, count=20):
    confusions = [
        (matrix[a, p], symbols[a], symbols[p])
        for a in range(len(symbols))
        for p in range(len(symbols))
        if a != p and matrix[a, p] > 0
    ]

    confusions.sort(reverse=True)

    print("\nTop confusions:")
    for errors, actual, predicted in confusions[:count]:
        print(f"{actual!r} -> {predicted!r}: {errors}")


# Arguments

def parse_args():
    parser = argparse.ArgumentParser(description="Train the writing CNN")

    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/writing_cnn.pt"))
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
    parser.add_argument("--tfds-data", type=Path, default=Path("data/tfds/parquet"))

    return parser.parse_args()


# Training

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    symbols = list(EMNIST_BYCLASS_CHARACTERS)
    number_of_classes = len(symbols)

    train_csv, test_csv = fold_paths(args.data, args.fold)

    print("Loading EMNIST ByClass through TFDS...")

    train_source = load_emnist_source(args.tfds_data, "train")
    test_source = load_emnist_source(args.tfds_data, "test")

    train_dataset = EMNISTDataset(train_source, augment=True)
    train_clean_dataset = EMNISTDataset(train_source, augment=False)
    test_dataset = EMNISTDataset(test_source, augment=False)

    persistent = args.workers > 0

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "persistent_workers": persistent,
    }

    if persistent:
        loader_args["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )

    train_clean_loader = DataLoader(
        train_clean_dataset,
        shuffle=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args,
    )

    model = WritingCNN(number_of_classes).to(device)
    loss_function = nn.CrossEntropyLoss()

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
    print(f"Classes: {number_of_classes}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Batch logging: every {args.log_every} batches")

    best_accuracy = 0.0
    epochs_without_improvement = 0

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

        clean_loss, clean_accuracy, _ = evaluate(
            model,
            train_clean_loader,
            loss_function,
            device,
            number_of_classes,
        )

        test_loss, test_accuracy, confusion = evaluate(
            model,
            test_loader,
            loss_function,
            device,
            number_of_classes,
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
            f"train_accuracy (clean)={clean_accuracy:.2%} "
            f"test_accuracy={test_accuracy:.2%} "
            f"lr={learning_rate:.2e} "
            f"time={elapsed:.1f}s "
            f"samples/s={samples_per_second:.0f}"
        )

        print_top_confusions(confusion, symbols)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "number_of_classes": number_of_classes,
                    "fold": args.fold,
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                },
                args.output,
            )

            print(f"  Saved best model to {args.output}")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience + 2:
                print("Early stopping.")
                break

    print(f"\nBest test accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()