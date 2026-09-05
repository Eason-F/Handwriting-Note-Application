from pathlib import Path
import shutil
import torch
from PIL import Image
import numpy as np

from writing_cnn.segmentation import LineSegmenter, WordSegmenter, CharacterSegmenter, ConjoinedCharacterSegmenter, SegmentationPipeline
from writing_cnn.data import EMNIST_BYCLASS_CHARACTERS
from writing_cnn.model import WritingCNN

IMAGE_PATH = "data/test_short.png"
CHECKPOINT_PATH = "checkpoints/writing_cnn.pt"

results_dir = Path("test_results")

if results_dir.exists():
    shutil.rmtree(results_dir)

results_dir.mkdir()

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

binary_image = Image.fromarray((~binary * 255).astype(np.uint8), mode="L")
binary_image.save(results_dir / "binary_test.png")

writing = SegmentationPipeline.find_writing_region(binary)

print(f"Writing region shape: {writing.shape}")

writing_image = Image.fromarray((~writing * 255).astype(np.uint8), mode="L")
writing_image.save(results_dir / "writing_test.png")

line_regions = line_segmenter.find_regions(writing)
lines = line_segmenter.crop(writing, line_regions)

print(f"\nNumber of lines: {len(lines)}")

for line_index, line in enumerate(lines):
    print(f"Line {line_index}: shape={line.shape}")

    line_dir = results_dir / f"line_{line_index}"
    line_dir.mkdir()

    line_image = Image.fromarray((~line * 255).astype(np.uint8), mode="L")
    line_image.save(line_dir / "line.png")

    word_regions = word_segmenter.find_regions(line)

    print(f"Number of words: {len(word_regions)}")

    words = word_segmenter.crop(line, word_regions)

    for word_index, word in enumerate(words):
        word_dir = line_dir / f"word_{word_index}"
        word_dir.mkdir()

        word_image = Image.fromarray((~word * 255).astype(np.uint8), mode="L")
        word_image.save(word_dir / "word.png")

        print(f"\nWord {word_index}: shape={word.shape}")

        character_regions = character_segmenter.find_regions(word)

        print(f"Initial character regions: {len(character_regions)}")

        conjoined_candidates = conjoined_segmenter.find_candidates(
            word,
            character_regions,
        )

        print(f"Conjoined candidates: {len(conjoined_candidates)}")

        processed_regions = conjoined_segmenter.process(
            word,
            character_regions,
            model,
            device,
        )

        print(f"Final character regions: {len(processed_regions)}")

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

            predicted_character = class_names[prediction] if 0 <= prediction < len(class_names) else "?"

            print(
                f"  Character {character_index}: "
                f"{predicted_character} "
                f"(confidence={confidence:.3f})"
            )