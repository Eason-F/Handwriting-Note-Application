"""Import extracted IAM word images without mixing writers across splits."""

import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip() and not line.lstrip().startswith('#'):
            yield number, line.split()


def writer_split(writer):
    bucket = int.from_bytes(hashlib.sha256(f'iam-v1:{writer}'.encode()).digest()[:4], 'big') % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def official_splits(directory):
    assignments = {}
    for split in ('train', 'validation', 'test'):
        for form in Path(directory, f'{split}.uttlist').read_text(encoding='utf-8').split():
            if form in assignments:
                raise ValueError(f'Form {form} occurs in multiple official splits')
            assignments[form] = split
    return assignments


def build_manifests(words_file, forms_file, images_root, splits_dir=None):
    writers = {}
    if forms_file:
        for number, fields in records(forms_file):
            if len(fields) < 2 or fields[0] in writers:
                raise ValueError(f'{forms_file}:{number}: invalid or duplicate form record')
            writers[fields[0]] = fields[1]

    split_by_form = official_splits(splits_dir) if splits_dir else None
    manifests = {split: [] for split in ('train', 'validation', 'test')}
    seen = set()
    rejected = 0
    for number, fields in records(words_file):
        if len(fields) < 9:
            raise ValueError(f'{words_file}:{number}: expected IAM word fields')
        word_id, status = fields[:2]
        if word_id in seen:
            raise ValueError(f'{words_file}:{number}: duplicate word {word_id}')
        seen.add(word_id)
        if status in {'er', 'err'}:
            rejected += 1
            continue
        if status != 'ok':
            raise ValueError(f'{words_file}:{number}: unknown segmentation status {status}')
        parts = word_id.split('-')
        if len(parts) != 4 or any(not part.isalnum() for part in parts):
            raise ValueError(f'{words_file}:{number}: invalid word ID {word_id}')
        form = '-'.join(parts[:2])
        if form not in writers and split_by_form is None:
            raise ValueError(f'Missing writer ID for form {form}')
        image = Path(images_root).resolve() / parts[0] / form / f'{word_id}.png'
        if not image.is_file():
            raise FileNotFoundError(image)
        writer = writers.get(form)
        split = split_by_form.get(form) if split_by_form is not None else writer_split(writer)
        if split is None:
            continue
        sample = {'image': str(image), 'text': ' '.join(fields[8:]), 'form_id': form,
                  'sample_id': word_id, 'dataset': 'IAM', 'unit': 'word'}
        if writer is not None:
            sample['writer_id'] = writer
        manifests[split].append(sample)
    if not any(manifests.values()):
        raise ValueError('No correctly segmented IAM words found')
    return manifests, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--words', type=Path, required=True, help='IAM words.txt')
    parser.add_argument('--forms', type=Path, help='IAM forms.txt, including writer IDs')
    parser.add_argument('--splits', type=Path, help='Directory containing official *.uttlist files')
    parser.add_argument('--images', type=Path, required=True, help='Extracted words image directory')
    parser.add_argument('--output', type=Path, required=True, help='New manifest directory')
    args = parser.parse_args()
    if bool(args.forms) == bool(args.splits):
        parser.error('provide exactly one of --forms or --splits')
    manifests, rejected = build_manifests(args.words, args.forms, args.images, args.splits)
    args.output.mkdir(parents=True, exist_ok=False)
    for split, samples in manifests.items():
        (args.output / f'{split}.json').write_text(json.dumps(samples, indent=2) + '\n', encoding='utf-8')
        identities = {sample.get('writer_id', sample['form_id']) for sample in samples}
        label = 'writers' if args.forms else 'forms'
        print(f'{split}: {len(samples)} words, {len(identities)} {label}')
    print(f'Excluded {rejected} words marked as segmentation errors.')


if __name__ == '__main__':
    main()
