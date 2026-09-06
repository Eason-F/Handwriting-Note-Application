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


image = Image.open(
    IMAGE_PATH
).convert("L")


segmented = segmentation_pipeline.segment(
    image,
    model,
    device,
)


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
                prediction_pipeline
                .predict_character_candidates(
                    model,
                    character,
                    device,
                    k=5,
                )
            )

            print(
                f"\n    Character "
                f"{character_index}:"
            )

            for rank, (
                prediction,
                confidence,
            ) in enumerate(
                candidates,
                start=1,
            ):
                print(
                    f"      {rank}. "
                    f"{prediction:<3} "
                    f"{confidence * 100:6.2f}%"
                )


print("\n" + "=" * 60)
print("WORD CANDIDATES")
print("=" * 60)


for line_index, line in enumerate(segmented):
    print(f"\nLine {line_index}:")

    for word_index, word in enumerate(line):
        candidates = (
            prediction_pipeline
            .predict_word_candidates(
                model,
                word,
                device,
                k=5,
                beam_width=10,
            )
        )

        reranked = (
            prediction_pipeline
            .rerank_word_candidates(
                candidates,
                frequency_weight=0.20,
                language_weight=0.35,
            )
        )

        print(
            f"\n  Word {word_index}:"
        )

        print("    CNN:")

        for rank, (
            text,
            score,
        ) in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"      {rank}. "
                f"{text:<20} "
                f"score={score:.4f}"
            )

        print("    Wordfreq + LM:")

        for rank, (
            text,
            score,
        ) in enumerate(
            reranked,
            start=1,
        ):
            frequency = (
                prediction_pipeline
                .word_frequency_score(
                    text
                )
            )

            language = (
                prediction_pipeline
                .language_score(
                    text
                )
            )

            print(
                f"      {rank}. "
                f"{text:<20} "
                f"score={score:.4f} "
                f"freq={frequency:.2f} "
                f"lm={language:.2f}"
            )


print("\n" + "=" * 60)
print("FINAL WORD-FREQUENCY + LM OUTPUT")
print("=" * 60)


final_lines = []

for line in segmented:
    final_words = []

    for word in line:
        candidates = (
            prediction_pipeline
            .predict_word_candidates(
                model,
                word,
                device,
                k=5,
                beam_width=10,
            )
        )

        reranked = (
            prediction_pipeline
            .rerank_word_candidates(
                candidates,
                frequency_weight=0.20,
                language_weight=0.35,
            )
        )

        if reranked:
            final_words.append(
                reranked[0][0]
            )
        elif candidates:
            final_words.append(
                candidates[0][0]
            )

    final_lines.append(
        " ".join(final_words)
    )


final_text = "\n".join(
    final_lines
)

print(final_text)


print("\n" + "=" * 60)
print("ORIGINAL CNN TOP-1 OUTPUT")
print("=" * 60)


predictions = prediction_pipeline.predict(
    segmented,
    model,
    device,
)

parsed_lines = []

for line_index, line in enumerate(predictions):
    parsed_line = " ".join(line)
    parsed_lines.append(parsed_line)

    print(
        f"Line {line_index}: "
        f"{parsed_line}"
    )


print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("\nCNN:")
print(
    "\n".join(parsed_lines)
)

print("\nWordfreq + LM:")
print(final_text)