import re
from pathlib import Path


class NoteStore:
    def __init__(self, root='notes'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.handwriting = self.root / '.handwriting'
        self.handwriting.mkdir(parents=True, exist_ok=True)

    def files(self):
        return sorted(self.root.glob('*.md'), key=lambda p: p.stat().st_mtime, reverse=True)

    def create(self, title='Untitled'):
        safe = re.sub(r'[^\w\- ]+', '', title).strip() or 'Untitled'
        path = self.root / f'{safe}.md'
        index = 2
        while path.exists():
            path = self.root / f'{safe} {index}.md'
            index += 1
        path.write_text(f'# {title}\n\n', encoding='utf-8')
        return path
