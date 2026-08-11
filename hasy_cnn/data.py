import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .images import prepare_symbol_image


@dataclass(frozen=True)
class SymbolInfo:
    symbol_id: int
    latex: str


def load_symbols(dataset_root: Path) -> list[SymbolInfo]:
    symbols_file = dataset_root / "symbols.csv"
    if not symbols_file.exists():
        raise FileNotFoundError(f"Missing HASY metadata: {symbols_file}")

    symbols: list[SymbolInfo] = []
    with symbols_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            symbols.append(SymbolInfo(int(row["symbol_id"]), row["latex"]))
    return sorted(symbols, key=lambda symbol: symbol.symbol_id)


class HASYDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        symbols: list[SymbolInfo],
        augment: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.augment = augment
        self.class_index = {
            symbol.symbol_id: index for index, symbol in enumerate(symbols)
        }
        self.samples: list[tuple[Path, int]] = []

        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                image_path = (csv_path.parent / row["path"]).resolve()
                symbol_id = int(row["symbol_id"])
                self.samples.append((image_path, self.class_index[symbol_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_index = self.samples[index]
        with Image.open(image_path) as image:
            tensor = prepare_symbol_image(image, augment=self.augment)
        return tensor, class_index


def fold_paths(dataset_root: Path, fold: int) -> tuple[Path, Path]:
    fold_root = dataset_root / "classification-task" / f"fold-{fold}"
    train_csv = fold_root / "train.csv"
    test_csv = fold_root / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Could not find fold {fold}. Expected {train_csv} and {test_csv}."
        )
    return train_csv, test_csv

