import argparse
from collections import defaultdict, deque
import json
import re
import unicodedata
from pathlib import Path

from PIL import Image


def normalize_text(text, strict=False):
    text = unicodedata.normalize('NFKC', text)
    if not strict:
        text = re.sub(r'[^\w\s]', ' ', text.casefold())
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


def error_counts(reference, hypothesis, words=False, strict=False):
    reference = normalize_text(reference, strict)
    hypothesis = normalize_text(hypothesis, strict)
    if words:
        reference, hypothesis = reference.split(), hypothesis.split()
    return edit_distance(reference, hypothesis), len(reference)


def format_rate(counts):
    errors, total = counts
    return f'{errors / total:.1%}' if total else f'n/a ({errors} insertions, empty reference)'


def balanced_sample(samples, limit=None):
    if limit is None or limit >= len(samples):
        return samples
    groups = defaultdict(deque)
    for sample in samples:
        groups[sample.get('writer_id', sample.get('form_id', 'unknown'))].append(sample)
    chosen = []
    while groups and len(chosen) < limit:
        for identity in sorted(tuple(groups)):
            chosen.append(groups[identity].popleft())
            if not groups[identity]:
                del groups[identity]
            if len(chosen) == limit:
                break
    return chosen


def benchmark(manifest_path, checkpoint_path=None, limit=None, verbose=True):
    from application.writing_recognition import HandwritingRecognizer

    manifest_path = Path(manifest_path)
    samples = balanced_sample(json.loads(manifest_path.read_text(encoding='utf-8')), limit)
    recognizer = HandwritingRecognizer(checkpoint_path) if checkpoint_path else HandwritingRecognizer()
    if not recognizer.ready:
        raise RuntimeError(recognizer.error or 'The writing CNN could not be loaded.')

    totals = {
        'characters': [0, 0], 'words': [0, 0], 'strict_characters': [0, 0],
        'raw_characters': [0, 0], 'raw_words': [0, 0],
    }
    for sample in samples:
        image_path = manifest_path.parent / sample['image']
        image = Image.open(image_path).convert('L')
        result = recognizer.recognize_word(image) if sample.get('unit') == 'word' else recognizer.recognize(image)
        character_counts = error_counts(sample['text'], result.text)
        word_counts = error_counts(sample['text'], result.text, words=True)
        raw_character_counts = error_counts(sample['text'], result.raw_text)
        raw_word_counts = error_counts(sample['text'], result.raw_text, words=True)
        strict_counts = error_counts(sample['text'], result.text, strict=True)
        totals['strict_characters'][0] += strict_counts[0]
        totals['strict_characters'][1] += strict_counts[1]
        totals['characters'][0] += character_counts[0]
        totals['characters'][1] += character_counts[1]
        totals['words'][0] += word_counts[0]
        totals['words'][1] += word_counts[1]
        totals['raw_characters'][0] += raw_character_counts[0]
        totals['raw_characters'][1] += raw_character_counts[1]
        totals['raw_words'][0] += raw_word_counts[0]
        totals['raw_words'][1] += raw_word_counts[1]

        if verbose:
            print(f"{sample['image']} ({result.elapsed_ms:.0f} ms)")
            print(f'  decoded CER: {format_rate(character_counts)}')
            print(f'  decoded WER: {format_rate(word_counts)}')
            print(f'  raw CER:     {format_rate(raw_character_counts)}')
            print(f'  raw WER:     {format_rate(raw_word_counts)}')
            print(f'  strict CER:  {format_rate(strict_counts)}')
            print(f'  output: {result.text.replace(chr(10), " / ")}')

    character_errors, characters = totals['characters']
    word_errors, words = totals['words']
    print(f'\nAggregate decoded CER: {format_rate((character_errors, characters))} ({character_errors}/{characters})')
    print(f'Aggregate decoded WER: {format_rate((word_errors, words))} ({word_errors}/{words})')
    print(f'Aggregate raw CER: {format_rate(totals["raw_characters"])} '
          f'({totals["raw_characters"][0]}/{totals["raw_characters"][1]})')
    print(f'Aggregate raw WER: {format_rate(totals["raw_words"])} '
          f'({totals["raw_words"][0]}/{totals["raw_words"][1]})')
    strict_errors, strict_characters = totals['strict_characters']
    print(f'Aggregate strict CER: {format_rate((strict_errors, strict_characters))} ({strict_errors}/{strict_characters})')


def main():
    parser = argparse.ArgumentParser(description='Benchmark end-to-end handwriting recognition accuracy.')
    parser.add_argument('manifest', type=Path, help='JSON manifest containing image paths and transcriptions.')
    parser.add_argument('--checkpoint', type=Path, help='Character-CNN checkpoint to evaluate.')
    parser.add_argument('--limit', type=int, help='Evaluate a deterministic identity-balanced subset.')
    parser.add_argument('--summary-only', action='store_true', help='Suppress per-sample output.')
    args = parser.parse_args()
    benchmark(args.manifest, args.checkpoint, args.limit, not args.summary_only)


if __name__ == '__main__':
    main()
