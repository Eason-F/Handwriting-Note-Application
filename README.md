# HASYv2 CNN symbol tester

This is the first neural-network component for the handwriting note application. A small PyTorch CNN learns to classify one isolated handwritten symbol from HASYv2. The PySide6 window lets you draw a symbol and inspect the model's five most likely predictions.

The default vocabulary excludes uncommon Greek letters and variants. It retains
the commonly useful lowercase symbols alpha, beta, gamma, delta, epsilon,
theta, lambda, mu, pi, rho, sigma, phi, and omega. This filtering happens while
the CSV files are loaded; it does not modify the original dataset.

It does **not** recognise complete expressions yet. The next stage would place this encoder inside a CNN-GRU-CTC expression recogniser.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and extract the official dataset:

```bash
mkdir -p data
curl -L --fail -o data/HASYv2.tar.bz2 \
  https://zenodo.org/records/259444/files/HASYv2.tar.bz2
tar -xjf data/HASYv2.tar.bz2 -C data
```

The expected metadata path is `data/symbols.csv`.

## Train

```bash
source .venv/bin/activate
python -m hasy_cnn.train --epochs 8
```

The trainer automatically uses Apple MPS when available and stores the best model at `checkpoints/hasy_cnn.pt`. For a quick smoke test, use `--epochs 1`.

The included local checkpoint was trained for eight epochs on fold 1 and reached
83.67% accuracy on its 16,992 held-out images. Checkpoints are intentionally
ignored by Git because they are generated files; rerunning the command recreates
one locally.

## Test in the GUI

```bash
source .venv/bin/activate
python -m hasy_cnn.gui
```

Draw one large, centred symbol at a time. The test canvas is cropped and resized to match HASYv2's 32x32 images.

## Run checks

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```
