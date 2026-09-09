import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .images import prepare_symbol_image, crop_and_center_drawing
from writing_cnn.data import EMNISTDataset

@dataclass(frozen=True)
class SymbolInfo:
    symbol_id: int
    latex: str
    samples: int
    


COMMON_GREEK_SYMBOLS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\theta",
    r"\lambda",
    r"\pi",
}

COMMON_MATHS_SYMBOLS = {
    r"+",
    r"-",
    r"\div",
    r"\times",
    r"\ast",
    r"\%",
    r"/",
    r"\equiv",
    r"\pi"
    r"\sqrt{}"
}

COMMON_MATH_ALPHANUMERIC = {"a", "b", "c", "x", "y", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}

def load_symbols(
    dataset_root: Path,
    allowed_greek: set[str] = COMMON_GREEK_SYMBOLS,
    allowed_maths: set[str] = COMMON_MATHS_SYMBOLS,
    allowed_alphanumeric: set[str] = COMMON_MATH_ALPHANUMERIC
) -> list[SymbolInfo]:
    symbols_file = dataset_root / "symbols.csv"
    if not symbols_file.exists():
        raise FileNotFoundError(f"Missing HASY metadata: {symbols_file}")

    allowed_symbols = allowed_greek | allowed_maths | allowed_alphanumeric
    symbols: list[SymbolInfo] = []
    with symbols_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["latex"] not in allowed_symbols:
                continue
            symbols.append(
                SymbolInfo(
                    int(row["symbol_id"]), 
                    str(row["latex"]), 
                    int(row["training_samples"]) + int(row["test_samples"])
                )
            )
    return sorted(symbols, key=lambda symbol: symbol.symbol_id)

class HASYDataset(Dataset):
    def __init__(self, csv_path: Path, symbols: list[SymbolInfo], augment: bool = False) -> None:
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
                if symbol_id in self.class_index:
                    self.samples.append((image_path, self.class_index[symbol_id]))
                    
    @staticmethod
    def get_total_class_samples(symbols) -> np.ndarray:
        return np.array([symbol.samples for symbol in symbols], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_index = self.samples[index]
        with Image.open(image_path) as image:
            image = crop_and_center_drawing(image)
            
            image = EMNISTDataset.augment_image(image) if self.augment else image
            tensor = prepare_symbol_image(image, False)
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
        

if __name__ == "__main__":
    print(load_symbols(Path("data")))