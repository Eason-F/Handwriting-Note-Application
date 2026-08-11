import csv
import tempfile
import unittest
from pathlib import Path

from hasy_cnn.data import COMMON_GREEK_SYMBOLS, load_symbols


class SymbolFilteringTests(unittest.TestCase):
    def test_common_greek_is_kept_and_uncommon_greek_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "symbols.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["symbol_id", "latex", "training_samples", "test_samples"]
                )
                writer.writerow([1, "A", 10, 2])
                writer.writerow([2, r"\pi", 10, 2])
                writer.writerow([3, r"\alpha", 10, 2])
                writer.writerow([4, r"\tau", 10, 2])
                writer.writerow([5, r"\varphi", 10, 2])

            symbols = load_symbols(root)
            labels = {symbol.latex for symbol in symbols}

        self.assertIn("A", labels)
        self.assertIn(r"\pi", labels)
        self.assertIn(r"\alpha", labels)
        self.assertNotIn(r"\tau", labels)
        self.assertNotIn(r"\varphi", labels)
        self.assertIn(r"\theta", COMMON_GREEK_SYMBOLS)


if __name__ == "__main__":
    unittest.main()

