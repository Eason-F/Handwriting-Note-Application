import argparse
import json
import re
import unicodedata
from pathlib import Path

from PIL import Image


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def error_counts(reference, hypothesis, words=False):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if words:
        reference, hypothesis = reference.split(), hypothesis.split()
    return edit_distance(reference, hypothesis), len(reference)


def benchmark(manifest_path):
    from application.writing_recognition import HandwritingRecognizer

    manifest_path = Path(manifest_path)
    samples = json.loads(manifest_path.read_text(encoding='utf-8'))
    recognizer = HandwritingRecognizer()
    if not recognizer.ready:
        raise RuntimeError(recognizer.error or 'The writing CNN could not be loaded.')

    totals = {'characters': [0, 0], 'words': [0, 0]}
    for sample in samples:
        image_path = manifest_path.parent / sample['image']
        result = recognizer.recognize(Image.open(image_path).convert('L'))
        character_counts = error_counts(sample['text'], result.text)
        word_counts = error_counts(sample['text'], result.text, words=True)
        raw_character_counts = error_counts(sample['text'], result.raw_text)
        raw_word_counts = error_counts(sample['text'], result.raw_text, words=True)
        totals['characters'][0] += character_counts[0]
        totals['characters'][1] += character_counts[1]
        totals['words'][0] += word_counts[0]
        totals['words'][1] += word_counts[1]

        print(f"{sample['image']} ({result.elapsed_ms:.0f} ms)")
        print(f'  decoded CER: {character_counts[0] / character_counts[1]:.1%}')
        print(f'  decoded WER: {word_counts[0] / word_counts[1]:.1%}')
        print(f'  raw CER:     {raw_character_counts[0] / raw_character_counts[1]:.1%}')
        print(f'  raw WER:     {raw_word_counts[0] / raw_word_counts[1]:.1%}')
        print(f'  output: {result.text.replace(chr(10), " / ")}')

    character_errors, characters = totals['characters']
    word_errors, words = totals['words']
    print(f'\nAggregate decoded CER: {character_errors / characters:.1%} ({character_errors}/{characters})')
    print(f'Aggregate decoded WER: {word_errors / words:.1%} ({word_errors}/{words})')


def main():
    parser = argparse.ArgumentParser(description='Benchmark end-to-end handwriting recognition accuracy.')
    parser.add_argument('manifest', type=Path, help='JSON manifest containing image paths and transcriptions.')
    benchmark(parser.parse_args().manifest)


if __name__ == '__main__':
    main()
