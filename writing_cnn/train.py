import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import MulticlassConfusionMatrix

from .data import EMNISTDataset, EMNIST_BYCLASS_CHARACTERS, fold_paths, load_emnist_source
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
    num_classes = model.classifier[-1].out_features
    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            scores = model(images)
            loss_total += loss_function(scores, targets).item() * targets.size(0)

            predictions = scores.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            sample_count += targets.size(0)

            confusion_matrix.update(predictions, targets)

    matrix = confusion_matrix.compute().cpu().numpy()

    return loss_total / sample_count, correct / sample_count, matrix


def print_class_accuracy(class_correct, class_total, symbols, columns=4):
    class_results = []

    for index, symbol in enumerate(symbols):
        total = int(class_total[index])
        correct = int(class_correct[index])
        accuracy = correct / total if total > 0 else 0.0
        class_results.append((symbol, accuracy, correct, total))

    symbol_width = 6
    accuracy_width = 9
    count_width = 14
    spacing = 3

    print("\nPer-class accuracy:")

    for _ in range(columns):
        print(
            f"{'class':>{symbol_width}} "
            f"{'accuracy':>{accuracy_width}} "
            f"{'correct/total':>{count_width}}",
            end=" " * spacing,
        )
    print()

    for _ in range(columns):
        print(
            f"{'-' * symbol_width} "
            f"{'-' * accuracy_width} "
            f"{'-' * count_width}",
            end=" " * spacing,
        )
    print()

    for row_start in range(0, len(class_results), columns):
        row = class_results[row_start:row_start + columns]

        for symbol, accuracy, correct, total in row:
            count_text = f"{correct}/{total}"
            print(
                f"{symbol!r:>{symbol_width}} "
                f"{accuracy:>{accuracy_width - 1}.2%} "
                f"{count_text:>{count_width}}",
                end=" " * spacing,
            )

        print()

def print_confusion_matrix(matrix, symbols):
    print("\nConfusion matrix:")
    print("Rows = actual, columns = predicted")

    width = max(3, max(len(symbol) for symbol in symbols) + 1)

    print(" " * width + " ".join(f"{symbol:>{width}}" for symbol in symbols))

    for index, symbol in enumerate(symbols):
        row = " ".join(f"{value:>{width}}" for value in matrix[index])
        print(f"{symbol:>{width}} {row}")

def print_top_confusions(matrix, symbols, count=20):
    confusions = []

    for actual in range(len(symbols)):
        for predicted in range(len(symbols)):
            if actual != predicted and matrix[actual, predicted] > 0:
                confusions.append(
                    (matrix[actual, predicted], symbols[actual], symbols[predicted])
                )

    confusions.sort(reverse=True)

    print("\nTop confusions:")
    for errors, actual, predicted in confusions[:count]:
        print(f"{actual!r} -> {predicted!r}: {errors}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the writing CNN")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/writing_cnn.pt"))
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tfds-data", type=Path, default=Path("data/tfds/parquet"))
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
        augment=False,
    )

    emnist_test = EMNISTDataset(
        load_emnist_source(args.tfds_data, "test"),
        augment=False,
    )

    sample_weights = torch.full(
        (len(emnist_train),),
        0.5 / len(emnist_train),
    )

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

    all_labels = np.array(
        [int(label) for image, label in emnist_train],
        dtype=np.int64,
    )

    classes, counts = np.unique(
        all_labels,
        return_counts=True,
    )

    num_classes = len(classes)
    total_samples = len(all_labels)

    weights = total_samples / (num_classes * counts)

    class_weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    model = WritingCNN(len(symbols)).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print(f"Classes: {len(symbols)}")
    print(f"EMNIST training samples: {len(emnist_train):,}")
    print(f"EMNIST test samples: {len(emnist_test):,}")
    print(f"Samples drawn per epoch: {len(sampler):,}")
    print(f"Initial learning rate: {args.learning_rate}")

    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()

        model.train()
        running_loss = 0.0
        train_correct = 0
        train_samples = 0

        for batch_number, (images, targets) in enumerate(emnist_train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)

            scores = model(images)
            loss = loss_function(scores, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            predictions = scores.argmax(dim=1)

            running_loss += loss.item() * targets.size(0)
            train_correct += (predictions == targets).sum().item()
            train_samples += targets.size(0)
            
            if batch_number % 100 == 0:
                print(
                    f"  epoch {epoch}: "
                    f"batch {batch_number}/{len(emnist_train_loader)} "
                    f"loss={running_loss / train_samples:.4f}",
                    flush=True,
                )

        train_loss = running_loss / train_samples
        train_accuracy = train_correct / train_samples
        
        test_loss, test_accuracy, confusion_matrix = evaluate(
            model,
            emnist_test_loader,
            loss_function,
            device,
        )

        scheduler.step(test_accuracy)
        current_lr = optimizer.param_groups[0]["lr"]
        seconds = time.perf_counter() - started

        print(
            f"\nEpoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"train_accuracy={train_accuracy:.2%} "
            f"test_accuracy={test_accuracy:.2%} "
            f"lr={current_lr:.6g} "
            f"time={seconds:.1f}s"
        )
        
        print_top_confusions(
            confusion_matrix,
            symbols,
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
                    "emnist_accuracy": test_accuracy,
                },
                args.output,
            )

            print(f"\n  Saved improved checkpoint to {args.output}")

    print(f"\nBest test accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()