# IAM evaluation

Keep IAM in the repository's top-level `data/iam` directory, separate from the
`writing_cnn` source package. Store `words.txt`, official split lists, and extracted word images in
`words/a01/a01-000u/a01-000u-00-00.png` layout.
Dataset source: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
Word annotation example: https://keras.io/examples/vision/handwriting_recognition/

Create a new directory of manifests:

```sh
IAM_DATA='data/iam'
python3 -m writing_cnn.iam --words "$IAM_DATA/IAM_Words/words.txt" --splits "$IAM_DATA/splits" --images "$IAM_DATA/words" --output "$IAM_DATA/manifests"
python3 -m writing_cnn.benchmark "$IAM_DATA/manifests/validation.json" --limit 500 --summary-only
```

With `--splits`, the importer uses IAM's official writer-independent form lists.
Alternatively, `--forms forms.txt` uses a deterministic writer hash for custom
80/10/10 splits. Paths in manifests are absolute and refer to the original
images. No images are copied or relabeled.

Use validation for development and reserve test for a frozen pipeline. Word
images assess character splitting and recognition, not page/line segmentation.
Word transcriptions are not character boundary annotations: do not assign their
letters to automatic character crops as if those crops were verified labels.
The importer rejects missing image paths and unknown writer IDs and excludes
records marked `err`. Image decoding is checked when the benchmark opens them.

Current status: IAM word images and official split lists are stored under
`data/iam`. Existing two-page metrics are development results and do not
establish performance on unseen IAM writers.
