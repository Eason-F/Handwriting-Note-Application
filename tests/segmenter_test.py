from pathlib import Path
import shutil

import torch
from PIL import Image

from writing_cnn.data import EMNIST_BYCLASS_CHARACTERS
from writing_cnn.model import WritingCNN
from writing_cnn.prediction import PredictionPipeline
from writing_cnn.segmentation import (
    CharacterSegmenter,
    ConjoinedCharacterSegmenter,
    LineSegmenter,
    SegmentationPipeline,
    WordSegmenter,
)


IMAGE_PATH = "data/test_short.png"
CHECKPOINT_PATH = "checkpoints/writing_cnn.pt"
RESULTS_DIR = Path("test_results")


if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)

RESULTS_DIR.mkdir()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model = WritingCNN(checkpoint["number_of_classes"])
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

line_segmenter = LineSegmenter(
    min_ink_pixels=8,
    max_internal_gap=2,
    min_line_height=10,
)

word_segmenter = WordSegmenter(
    word_gap_threshold=8,
)

character_segmenter = CharacterSegmenter()

conjoined_segmenter = ConjoinedCharacterSegmenter(
    width_multiplier=1.8,
    min_character_width=5,
)

segmentation_pipeline = SegmentationPipeline(
    line_segmenter,
    word_segmenter,
    character_segmenter,
    conjoined_segmenter,
)

prediction_pipeline = PredictionPipeline(
    list(EMNIST_BYCLASS_CHARACTERS)
)

image = Image.open(IMAGE_PATH).convert("L")

segmented = segmentation_pipeline.segment(
    image,
    model,
    device,
)

for line_index, line in enumerate(segmented):
    line_dir = RESULTS_DIR / f"line_{line_index}"
    line_dir.mkdir()

    print(f"\nLine {line_index}: {len(line)} words")

    for word_index, word in enumerate(line):
        word_dir = line_dir / f"word_{word_index}"
        word_dir.mkdir()

        print(f"  Word {word_index}: {len(word)} characters")

        for character_index, character in enumerate(word):
            character.save(
                word_dir / f"character_{character_index}.png"
            )

predictions = prediction_pipeline.predict(
    segmented,
    model,
    device,
)

parsed_lines = []

print("\nPredictions:")

for line_index, line in enumerate(predictions):
    parsed_line = " ".join(line)
    parsed_lines.append(parsed_line)

    print(f"  Line {line_index}: {parsed_line}")

parsed_text = "\n".join(parsed_lines)

print("\nParsed output:")
print(parsed_text)

