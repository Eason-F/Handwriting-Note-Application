import argparse
from pathlib import Path

import torch
import time
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from .data import ContextualEMNISTDataset, EMNISTDataset, EMNIST_BYCLASS_CHARACTERS, load_emnist_source
from .model import WritingCNN


def choose_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate(model, loader, device):
    exact = folded = top_three = samples = 0
    symbols = EMNIST_BYCLASS_CHARACTERS
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            scores = model(images)
            predictions = scores.argmax(dim=1)
            top = scores.topk(3, dim=1).indices
            exact += (predictions == targets).sum().item()
            top_three += (top == targets[:, None]).any(dim=1).sum().item()
            folded += sum(
                symbols[prediction].casefold() == symbols[target].casefold()
                for prediction, target in zip(predictions.tolist(), targets.tolist())
            )
            samples += len(targets)
    return exact / samples, folded / samples, top_three / samples


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune the character CNN on real glyphs in word context.')
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/writing_cnn.pt'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/writing_cnn_candidate.pt'))
    parser.add_argument('--tfds-data', type=Path, default=Path('data/tfds/parquet'))
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--learning-rate', type=float, default=8e-5)
    parser.add_argument('--context-samples', type=int, default=23200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)

    train_source = load_emnist_source(args.tfds_data, 'train')
    test_source = load_emnist_source(args.tfds_data, 'test')
    training = ConcatDataset((
        EMNISTDataset(train_source, augment=True),
        ContextualEMNISTDataset(train_source, args.context_samples),
    ))
    validation = EMNISTDataset(test_source, augment=False)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = WritingCNN(checkpoint['number_of_classes']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_case_folded = 0.0
    
    print(f"Device: {device}")
    print(f"Training samples: {len(training):,}")
    print(f"Test samples: {len(validation):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Learning rate: {args.learning_rate}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_total = samples = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(targets)
            samples += len(targets)

        exact, case_folded, top_three = evaluate(model, validation_loader, device)
        print(
            f'Epoch {epoch}/{args.epochs}: loss={loss_total / samples:.4f}, '
            f'exact={exact:.2%}, case-folded={case_folded:.2%}, top-3={top_three:.2%}',
            f'time={(time.perf_counter() - started):.2f}',
            flush=True,
        )
        if case_folded > best_case_folded:
            best_case_folded = case_folded
            torch.save({
                **{key: value for key, value in checkpoint.items() if key != 'model_state'},
                'model_state': model.state_dict(),
                'contextual_emnist_samples': args.context_samples,
                'validation_accuracy': exact,
                'validation_case_folded_accuracy': case_folded,
                'validation_top3_accuracy': top_three,
            }, args.output)
            print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
