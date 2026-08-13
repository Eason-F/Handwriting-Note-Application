import csv
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
import matplotlib._mathtext_data as mtd

from .images import prepare_symbol_image


GREEK_SYMBOLS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\zeta",
    r"\eta",
    r"\theta",
    r"\Theta",
    r"\epsilon",
    r"\varepsilon",
    r"\iota",
    r"\kappa",
    r"\varkappa",
    r"\lambda",
    r"\Lambda",
    r"\mu",
    r"\nu",
    r"\xi",
    r"\Xi",
    r"\pi",
    r"\Pi",
    r"\rho",
    r"\varrho",
    r"\sigma",
    r"\Sigma",
    r"\tau",
    r"\phi",
    r"\Phi",
    r"\varphi",
    r"\chi",
    r"\psi",
    r"\Psi",
    r"\omega",
    r"\Omega",
}

COMMON_GREEK_SYMBOLS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\theta",
    r"\lambda",
    r"\mu",
    r"\pi",
    r"\rho",
    r"\sigma",
    r"\phi",
    r"\omega",
}

COMMON_MATHS_SYMBOLS = {
    r"+",
    r"-",
    r"\div",
    r"\times",
    r"\ast",
    r"\%",
    r"/",
    r"\equiv"
}

@dataclass(frozen=True)
class SymbolInfo:
    symbol_id: int
    latex: str


EMNIST_BYCLASS_CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase

def load_symbols(
    dataset_root: Path,
    allowed_greek: set[str] = COMMON_GREEK_SYMBOLS,
    allowed_maths: set[str] = COMMON_MATHS_SYMBOLS,
    compacted: bool = False
) -> list[SymbolInfo]:
    symbols_file = dataset_root / "symbols.csv"
    if not symbols_file.exists():
        raise FileNotFoundError(f"Missing HASY metadata: {symbols_file}")

    symbols: list[SymbolInfo] = []
    with symbols_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            latex = row["latex"]
            if latex in GREEK_SYMBOLS and latex not in allowed_greek:
                continue
            elif latex not in allowed_maths and latex not in allowed_greek and compacted:
                continue
            symbols.append(SymbolInfo(int(row["symbol_id"]), latex))
    return sorted(symbols, key=lambda symbol: symbol.symbol_id)


def add_emnist_characters(symbols: list[SymbolInfo]) -> list[SymbolInfo]:
    """Ensure every EMNIST ByClass character exists in the shared vocabulary."""
    extended = list(symbols)
    existing = {symbol.latex for symbol in extended}
    next_id = max(symbol.symbol_id for symbol in extended) + 1
    for character in EMNIST_BYCLASS_CHARACTERS:
        if character not in existing:
            extended.append(SymbolInfo(next_id, character))
            existing.add(character)
            next_id += 1
    return extended


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
                if symbol_id in self.class_index:
                    self.samples.append((image_path, self.class_index[symbol_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_index = self.samples[index]
        with Image.open(image_path) as image:
            tensor = prepare_symbol_image(image, augment=self.augment)
        return tensor, class_index


class EMNISTDataset(Dataset):
    """Adapt a TFDS EMNIST source to the same output format as HASYDataset."""

    def __init__(self, source, symbols: list[SymbolInfo], augment: bool = False) -> None:
        self.source = source
        self.augment = augment
        class_index = {symbol.latex: index for index, symbol in enumerate(symbols)}
        missing = set(EMNIST_BYCLASS_CHARACTERS) - class_index.keys()
        if missing:
            raise ValueError(f"Shared vocabulary is missing EMNIST characters: {missing}")
        self.label_to_class = [
            class_index[character] for character in EMNIST_BYCLASS_CHARACTERS
        ]

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        example = self.source[index]
        image_array = np.asarray(example["image"]).squeeze().T
        # EMNIST is white ink on black; HASY and the GUI use black ink on white.
        image = ImageOps.invert(Image.fromarray(image_array.astype(np.uint8), mode="L"))
        label = int(example["label"])
        tensor = prepare_symbol_image(image, augment=self.augment)
        return tensor, self.label_to_class[label]


def load_emnist_source(data_dir: Path, split: str):
    """Download if needed and return a TensorFlow-free TFDS data source."""
    import tensorflow_datasets as tfds

    return tfds.data_source(
        "emnist/byclass",
        split=split,
        data_dir=data_dir,
        download=True,
        builder_kwargs={"file_format": "parquet"},
        download_and_prepare_kwargs={"download_dir": data_dir.parent / "downloads"},
    )


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