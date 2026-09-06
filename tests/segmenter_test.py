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


# Paths

IMAGE_PATH = "data/test_short.png"
CHECKPOINT_PATH = "checkpoints/writing_cnn.pt"
RESULTS_DIR = Path("test_results")


# Setup

if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)

RESULTS_DIR.mkdir()

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Device: {device}")


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model = WritingCNN(
    checkpoint["number_of_classes"]
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.to(device)
model.eval()


# Segmentation

line_segmenter = LineSegmenter(
    min_ink_pixels=8,
    max_internal_gap=2,
    min_line_height=10,
)

word_segmenter = WordSegmenter()
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


# Segmentation output

print("\n" + "=" * 60)
print("SEGMENTATION")
print("=" * 60)

for line_index, line in enumerate(segmented):
    print(
        f"\nLine {line_index}: "
        f"{len(line)} words"
    )

    line_dir = RESULTS_DIR / f"line_{line_index}"
    line_dir.mkdir()

    for word_index, word in enumerate(line):
        print(
            f"  Word {word_index}: "
            f"{len(word)} characters"
        )

        word_dir = line_dir / f"word_{word_index}"
        word_dir.mkdir()

        for character_index, character in enumerate(word):
            character.save(
                word_dir /
                f"character_{character_index}.png"
            )


# CNN character predictions

print("\n" + "=" * 60)
print("CNN TOP-5 PREDICTIONS")
print("=" * 60)

for line_index, line in enumerate(segmented):
    print(f"\nLine {line_index}:")

    for word_index, word in enumerate(line):
        print(
            f"\n  Word {word_index}: "
            f"{len(word)} characters"
        )

        for character_index, character in enumerate(word):
            candidates = (
                prediction_pipeline.predict_character_candidates(
                    model,
                    character,
                    device,
                    k=5,
                )
            )

            print(f"\n    Character {character_index}:")

            for rank, (prediction, confidence) in enumerate(
                candidates,
                start=1,
            ):
                print(
                    f"      {rank}. "
                    f"{prediction:<3} "
                    f"{confidence * 100:6.2f}%"
                )


# Word prediction methods

methods = (
    "cnn",
    "wordfreq",
    "hybrid",
    "wordfreq_hybrid",
)

print("\n" + "=" * 60)
print("WORD CANDIDATES")
print("=" * 60)

for line_index, line in enumerate(segmented):
    print(f"\nLine {line_index}:")

    for word_index, word in enumerate(line):
        print(
            f"\n  Word {word_index}: "
            f"{len(word)} characters"
        )

        for method in methods:
            candidates = prediction_pipeline.predict_word(
                model,
                word,
                device,
                method=method,
                character_top_k=5,
                beam_width=10,
                frequency_weight=0.20,
                lm_weight=0.35,
                limit=5,
            )

            print(f"\n    {method}:")

            for rank, (text, score) in enumerate(
                candidates,
                start=1,
            ):
                print(
                    f"      {rank}. "
                    f"{text:<20} "
                    f"score={score:.4f}"
                )


# Wordfreq + LM details

print("\n" + "=" * 60)
print("WORD DETAILS")
print("=" * 60)

for line_index, line in enumerate(segmented):
    print(f"\nLine {line_index}:")

    for word_index, word in enumerate(line):
        candidates = prediction_pipeline.predict_word(
            model,
            word,
            device,
            method="wordfreq_hybrid",
            character_top_k=5,
            beam_width=10,
            frequency_weight=0.20,
            lm_weight=0.35,
            limit=5,
        )

        print(f"\n  Word {word_index}:")

        for rank, (text, score) in enumerate(
            candidates,
            start=1,
        ):
            frequency = prediction_pipeline.wordfreq_score(text)
            language = prediction_pipeline.lm_score(text)

            print(
                f"    {rank}. "
                f"{text:<20} "
                f"score={score:.4f} "
                f"freq={frequency:.2f} "
                f"lm={language:.2f}"
            )


# Final outputs

print("\n" + "=" * 60)
print("FINAL OUTPUTS")
print("=" * 60)

outputs = {method: [] for method in methods}

for line in segmented:
    for method in methods:
        outputs[method].append([])

    for word in line:
        for method in methods:
            candidates = prediction_pipeline.predict_word(
                model,
                word,
                device,
                method=method,
                character_top_k=5,
                beam_width=10,
                frequency_weight=0.20,
                lm_weight=0.35,
                limit=5,
            )

            if candidates:
                outputs[method][-1].append(
                    candidates[0][0]
                )
            else:
                outputs[method][-1].append("")


for method in methods:
    outputs[method] = [
        " ".join(line)
        for line in outputs[method]
    ]


print("\nCNN:")
print("\n".join(outputs["cnn"]))

print("\nWordFreq:")
print("\n".join(outputs["wordfreq"]))

print("\nHybrid:")
print("\n".join(outputs["hybrid"]))

print("\nWordFreq + LM:")
print("\n".join(outputs["wordfreq_hybrid"]))


# Direct pipeline output

print("\n" + "=" * 60)
print("PREDICTION PIPELINE OUTPUT")
print("=" * 60)

predictions = prediction_pipeline.predict(
    segmented,
    model,
    device,
    method="wordfreq_hybrid",
    character_top_k=5,
    beam_width=10,
    frequency_weight=0.20,
    lm_weight=0.35,
)

parsed_lines = [
    " ".join(line)
    for line in predictions
]

print("\n".join(parsed_lines))


# Comparison

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("\nCNN:")
print("\n".join(outputs["cnn"]))

print("\nWordFreq:")
print("\n".join(outputs["wordfreq"]))

print("\nHybrid:")
print("\n".join(outputs["hybrid"]))

print("\nWordFreq + LM:")
print("\n".join(outputs["wordfreq_hybrid"]))