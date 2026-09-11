# InkNote — Handwriting Note Application

InkNote is a desktop note-taking application that combines Markdown-style text notes, CNN handwriting recognition, single-symbol mathematical handwriting recognition, and symbolic expression evaluation. It is built with Python, PySide6, PyTorch, Pillow, and SymPy.

## Main features

- Create, open, rename, duplicate, move, and delete notes and folders.
- Save notes normally or use Save As to create a copy elsewhere.
- Browse the note library from the left sidebar.
- Type and edit notes with normal undo, redo, cut, copy, paste, and selection shortcuts.
- Open a resizable handwriting panel inside the main window.
- Recognise multiline text handwriting with the writing CNN and segmentation pipeline.
- Build mathematical expressions one handwritten symbol at a time with the math CNN.
- Evaluate numeric and symbolic expressions directly from the note.
- Keep independent SymPy variable assignments for each open note during the session.
- Insert handwriting as an image when the original ink should be preserved in the note.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- `checkpoints/writing_cnn.pt`
- `checkpoints/math_cnn.pt`

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Running the application

From the repository root, run:

```bash
python main.py
```

Notes are stored in the repository's `notes` directory by default. Handwriting images inserted into notes are stored in `notes/.handwriting`.

## Working with notes

Use the buttons below the sidebar to create notes and folders or refresh the library. Double-click a note to open it. Right-click a sidebar item to open, rename, duplicate, move, or delete it. The context menu can also reveal the note-storage directory.

InkNote autosaves changed documents. It also checks for unsaved changes before switching notes or closing, so a failed save or cancelled prompt does not silently discard work.

Common shortcuts include:

- `Ctrl+N`: new note
- `Ctrl+O`: open a note
- `Ctrl+S`: save
- `Ctrl+Shift+S`: Save As
- `Ctrl+F`: focus the note filter
- `Ctrl+Shift+H`: toggle the handwriting panel
- `Ctrl+=`: evaluate the selection or current line
- `Ctrl+Shift+=`: evaluate and insert a result

Shortcuts can be changed from the Settings dialog.

## Text handwriting

1. Ensure the mode button says **Text**.
2. Click **Handwriting input** to expand the bottom panel.
3. Drag the splitter above the panel to adjust the canvas height for multiline writing.
4. Write on the canvas with the mouse, stylus, or Trackpad Draw option.
5. Pause for approximately 1.2 seconds to generate an automatic recognition preview.
6. Correct the preview if necessary, then click **Recognise & insert**.

Recognised text is inserted at the current editor cursor. Line breaks from multiline handwriting are preserved. The canvas is not automatically discarded after insertion. Use **Clear** when ready to begin another passage, or **Keep as ink** to insert the canvas as an image.

## Mathematical handwriting

Click the **Text** mode button so that it changes to **Math**. The canvas becomes a centred square because `math_cnn.pt` recognises one symbol at a time.

1. Draw one symbol and pause.
2. The recognised symbol is appended to the expression shown below the canvas.
3. The symbol canvas clears automatically so the next symbol can be written.
4. Repeat until the expression is complete.
5. Click **Recognise & insert** while the symbol canvas is empty to insert the accumulated expression at the editor cursor.
6. Use **Clear** to discard the accumulated expression and current symbol.

The math recogniser intentionally remains a single-symbol CNN. Expression assembly is handled by the application instead of asking the CNN to segment a complete equation.

Greek predictions are displayed and inserted as their actual glyphs, including `π`, `α`, `β`, `γ`, `δ`, `θ`, `ε`, and `λ`. Exact irrational results retain their symbolic form and include a rounded value beside it, such as `sqrt(2) [≈ 1.41421]`.

## Evaluating mathematics

Select an expression in the editor or place the cursor on its line, then choose **Evaluate**. InkNote writes the answer into the document, for example:

```text
5 * 5 = 25
```

The updated line is saved to the current note file immediately; it is not only shown in the status bar or left waiting for autosave.

Supported input includes arithmetic, variables, implicit multiplication, common functions, expansion, factoring, simplification, and equation solving:

```text
x = 5
y = x + 2
x * y
2x + 3x
expand((x + 1)^2)
factor(x^2 - 1)
solve(x^2 - 4, x)
5x = 10 x=?
```

The final example produces `x = 2`. Unicode handwriting substitutions such as `×`, `÷`, `−`, and `π` are normalised by the math layer. Invalid or undefined expressions are reported in the status bar rather than crashing the application.

## Project structure

```text
main.py                         Application entry point
application/main_window.py      Main window and workflow coordination
application/handwriting_panel.py Embedded handwriting controls and canvas
application/canvas.py           Drawing and image conversion
application/notes.py            NoteDocument and NoteManager
application/calculator.py       Stateful SymPy evaluation
application/writing_recognition.py Writing-model integration and worker
application/math_recognition.py Single-symbol math-model integration
writing_cnn/                    Text CNN, segmentation, and prediction pipeline
math_cnn/                       Mathematical-symbol CNN and image processing
checkpoints/                    Trained model checkpoints
tests/                          Automated tests and recognition checks
```

## Tests

Run the automated test suite from the repository root:

```bash
python -m unittest discover -v
```

The recognition models can be large, so GUI startup may take a moment while both checkpoints are loaded.
