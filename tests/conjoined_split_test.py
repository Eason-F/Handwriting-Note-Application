from pathlib import Path
import shutil

import torch
from PIL import Image
import numpy as np

from writing_cnn.segmentation import (
    LineSegmenter,
    WordSegmenter,
    CharacterSegmenter,
    ConjoinedCharacterSegmenter,
    SegmentationPipeline,
)
from writing_cnn.data import EMNIST_BYCLASS_CHARACTERS
from writing_cnn.model import WritingCNN


IMAGE_PATH = "data/test_short.png"
CHECKPOINT_PATH = "checkpoints/writing_cnn.pt"

results_dir = Path("test_results")

if results_dir.exists():
    shutil.rmtree(results_dir)

results_dir.mkdir()

line_segmenter = LineSegmenter()
word_segmenter = WordSegmenter()
character_segmenter = CharacterSegmenter()
conjoined_segmenter = ConjoinedCharacterSegmenter()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

model = WritingCNN(checkpoint["number_of_classes"])
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

class_names = list(EMNIST_BYCLASS_CHARACTERS)

image = Image.open(IMAGE_PATH).convert("L")
print(f"Original image size: {image.size}")

binary = SegmentationPipeline.binarize(image)

print(f"Binary shape: {binary.shape}")
print(f"Handwriting pixels: {np.sum(binary)}")
print(f"Foreground percentage: {100 * np.mean(binary):.2f}%")

Image.fromarray(
    (~binary * 255).astype(np.uint8),
    mode="L",
).save(results_dir / "binary_test.png")

writing = SegmentationPipeline.find_writing_region(binary)
print(f"Writing region shape: {writing.shape}")

Image.fromarray(
    (~writing * 255).astype(np.uint8),
    mode="L",
).save(results_dir / "writing_test.png")

line_regions = line_segmenter.find_regions(writing)
lines = line_segmenter.crop(writing, line_regions)

print(f"\nNumber of lines: {len(lines)}")

for line_index, line in enumerate(lines):
    print(f"Line {line_index}: shape={line.shape}")

    line_dir = results_dir / f"line_{line_index}"
    line_dir.mkdir()

    Image.fromarray(
        (~line * 255).astype(np.uint8),
        mode="L",
    ).save(line_dir / "line.png")

    word_regions = word_segmenter.find_regions(line)
    words = word_segmenter.crop(line, word_regions)

    print(f"Number of words: {len(word_regions)}")

    for word_index, word in enumerate(words):
        word_dir = line_dir / f"word_{word_index}"
        word_dir.mkdir()

        Image.fromarray(
            (~word * 255).astype(np.uint8),
            mode="L",
        ).save(word_dir / "word.png")

        print(f"\nWord {word_index}: shape={word.shape}")

        character_regions = character_segmenter.find_regions(word)

        print(f"Initial character regions: {len(character_regions)}")
        print(f"Initial regions: {character_regions}")

        conjoined_candidates = conjoined_segmenter.find_candidates(
            word,
            character_regions,
        )

        print(f"Conjoined candidates: {len(conjoined_candidates)}")
        print(f"Conjoined regions: {conjoined_candidates}")

        processed_regions = conjoined_segmenter.process(
            word,
            character_regions,
            model,
            device,
        )

        print(f"Final character regions: {len(processed_regions)}")
        print(f"Final regions: {processed_regions}")

        characters = character_segmenter.crop(
            word,
            processed_regions,
        )

        for character_index, character in enumerate(characters):
            character.save(
                word_dir / f"character_{character_index}.png"
            )

            prediction, confidence = conjoined_segmenter.predict(
                model,
                character,
                device,
            )

            predicted_character = (
                class_names[prediction]
                if 0 <= prediction < len(class_names)
                else "?"
            )

            print(
                f"  Character {character_index}: "
                f"{predicted_character} "
                f"(confidence={confidence:.3f})"
            )