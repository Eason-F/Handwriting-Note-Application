from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class NoteError(Exception):
    """A recoverable note-storage error suitable for displaying in the GUI."""


@dataclass
class NoteDocument:
    path: Path | None = None
    text: str = ''
    strokes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False

    @property
    def title(self):
        return self.path.stem if self.path else 'Untitled'

    def mark_dirty(self):
        self.dirty = True


class NoteManager:
    """Owns all filesystem operations for notes and their ink sidecars."""

    INVALID_NAMES = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path.home() / 'Documents' / 'InkNote Notes' if getattr(sys, 'frozen', False) else Path.cwd() / 'notes'
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.handwriting = self.root / '.handwriting'
        self.handwriting.mkdir(exist_ok=True)

    def _inside_root(self, path: Path):
        try:
            path.resolve().relative_to(self.root)
        except ValueError as exc:
            raise NoteError('The selected location is outside the note library.') from exc
        return path

    def validate_name(self, name: str):
        name = name.strip()
        if not name or name in {'.', '..'} or self.INVALID_NAMES.search(name):
            raise NoteError('Use a non-empty name without /, \\, :, *, ?, quotes, or angle brackets.')
        return name

    def unique_path(self, folder: Path, name: str, suffix: str = '.md'):
        name = self.validate_name(name)
        base = Path(name).stem if Path(name).suffix.lower() in {'.md', '.txt'} else name
        path = folder / f'{base}{suffix}'
        index = 2
        while path.exists():
            path = folder / f'{base} {index}{suffix}'
            index += 1
        return path

    def list_entries(self, folder: Path | None = None):
        folder = self._inside_root(folder or self.root)
        if not folder.is_dir():
            raise NoteError(f'Folder no longer exists: {folder.name}')
        return sorted(
            (p for p in folder.iterdir() if not p.name.startswith('.') and (p.is_dir() or p.suffix.lower() in {'.md', '.txt'})),
            key=lambda p: (not p.is_dir(), p.name.casefold()),
        )

    def files(self):
        return sorted(p for p in self.root.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(self.root).parts) and p.suffix.lower() in {'.md', '.txt'})

    def create_note(self, name='Untitled', folder: Path | None = None):
        folder = self._inside_root(folder or self.root)
        folder.mkdir(parents=True, exist_ok=True)
        path = self.unique_path(folder, name)
        document = NoteDocument(path=path, text=f'# {path.stem}\n\n', dirty=True)
        self.save_note(document)
        return document

    def create(self, title='Untitled'):
        return self.create_note(title).path

    def create_folder(self, name: str, parent: Path | None = None):
        parent = self._inside_root(parent or self.root)
        path = parent / self.validate_name(name)
        if path.exists():
            raise NoteError(f'“{path.name}” already exists.')
        path.mkdir()
        return path

    def open_note(self, path: str | Path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise NoteError(f'Note no longer exists: {path.name}')
        if not path.is_file() or path.suffix.lower() not in {'.md', '.txt'}:
            raise NoteError('Only Markdown and text note files can be opened.')
        strokes, metadata = [], {}
        sidecar = self.sidecar_path(path)
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding='utf-8'))
                strokes = data.get('strokes', [])
                metadata = data.get('metadata', {})
            except (OSError, ValueError, TypeError) as exc:
                raise NoteError(f'Could not read handwriting data for {path.name}: {exc}') from exc
        try:
            text = path.read_text(encoding='utf-8')
        except OSError as exc:
            raise NoteError(f'Could not open {path.name}: {exc}') from exc
        return NoteDocument(path=path, text=text, strokes=strokes, metadata=metadata)

    def save_note(self, document: NoteDocument, path: str | Path | None = None, overwrite=False):
        target = Path(path).expanduser().resolve() if path else document.path
        if target is None:
            raise NoteError('Choose a filename before saving this note.')
        if target.suffix.lower() not in {'.md', '.txt'}:
            target = target.with_suffix('.md')
        if path and target.exists() and target != document.path and not overwrite:
            raise NoteError(f'“{target.name}” already exists. Choose another name.')
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(document.text, encoding='utf-8')
            payload = {'version': 1, 'strokes': document.strokes, 'metadata': document.metadata}
            self.sidecar_path(target).write_text(json.dumps(payload, indent=2), encoding='utf-8')
        except OSError as exc:
            raise NoteError(f'Could not save {target.name}: {exc}') from exc
        document.path = target
        document.dirty = False
        return target

    def rename_note(self, path: Path, new_name: str):
        path = self._inside_root(Path(path))
        suffix = path.suffix if path.is_file() else ''
        target = path.with_name(self.validate_name(new_name) + suffix)
        if target.exists():
            raise NoteError(f'“{target.name}” already exists.')
        old_sidecar = self.sidecar_path(path) if path.is_file() else None
        path.rename(target)
        if old_sidecar and old_sidecar.exists():
            old_sidecar.rename(self.sidecar_path(target))
        return target

    def delete_note(self, path: Path):
        path = self._inside_root(Path(path))
        if path == self.root:
            raise NoteError('The note library cannot be deleted.')
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
            self.sidecar_path(path).unlink(missing_ok=True)
        else:
            raise NoteError(f'Item no longer exists: {path.name}')

    def duplicate_note(self, path: Path):
        source = self.open_note(path)
        target = self.unique_path(path.parent, f'{path.stem} copy', path.suffix)
        copy = NoteDocument(target, source.text, list(source.strokes), dict(source.metadata), True)
        self.save_note(copy)
        return target

    def move_note(self, path: Path, folder: Path):
        path, folder = self._inside_root(Path(path)), self._inside_root(Path(folder))
        if not folder.is_dir():
            raise NoteError('Choose a folder as the destination.')
        target = folder / path.name
        if target.exists():
            raise NoteError(f'“{path.name}” already exists in {folder.name}.')
        old_sidecar = self.sidecar_path(path)
        path.rename(target)
        if old_sidecar.exists():
            old_sidecar.rename(self.sidecar_path(target))
        return target

    @staticmethod
    def sidecar_path(path: Path):
        return path.with_name(f'.{path.name}.inknote.json')


NoteStore = NoteManager
