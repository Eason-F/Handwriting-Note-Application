import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
except ImportError:
    torch = None

from math_cnn.images import prepare_symbol_image
from math_cnn.model import SymbolCNN

from .theme import MATH_CHECKPOINT_PATH


@dataclass
class MathResult:
    expression: str
    answer: str
    elapsed_ms: float
    symbols: int


class MathRecognizer:
    replacements = {
        r'\\times': '*', r'\\ast': '*', r'\\div': '/', r'\\cdot': '*',
        r'\\minus': '-', r'\\plus': '+', r'\\%': '%', r'\\equiv': '=',
        r'\\sqrt\{\}': 'sqrt',
    }

    def __init__(self):
        self.device = torch.device('mps' if torch and torch.backends.mps.is_available() else 'cpu') if torch else None
        self.model = None
        self.symbols = None
        self.error = None
        if torch is None:
            self.error = 'PyTorch is not installed.'
            return
        try:
            checkpoint = torch.load(Path(MATH_CHECKPOINT_PATH), map_location=self.device, weights_only=True)
            self.symbols = checkpoint['symbols']
            self.model = SymbolCNN(checkpoint['number_of_classes'])
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device).eval()
        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self):
        return self.model is not None and self.symbols is not None

    @staticmethod
    def _components(image):
        gray = np.asarray(image.convert('L'))
        mask = gray < 220
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                xs, ys = [], []
                while stack:
                    cy, cx = stack.pop()
                    xs.append(cx); ys.append(cy)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(xs) >= 10:
                    components.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
        return components

    @staticmethod
    def _merge_components(boxes):
        boxes = boxes[:]
        changed = True
        while changed:
            changed = False
            result = []
            used = [False] * len(boxes)
            for i, box in enumerate(boxes):
                if used[i]:
                    continue
                current = box
                used[i] = True
                for j, other in enumerate(boxes):
                    if used[j]:
                        continue
                    ax1, ay1, ax2, ay2 = current
                    bx1, by1, bx2, by2 = other
                    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                    min_width = max(1, min(ax2 - ax1, bx2 - bx1))
                    vertical_gap = max(0, max(by1, ay1) - min(by2, ay2))
                    if overlap / min_width > 0.5 and vertical_gap <= 18:
                        current = (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
                        used[j] = True
                        changed = True
                result.append(current)
            boxes = result
        return sorted(boxes, key=lambda b: (b[0], b[1]))

    def _segment(self, image):
        grayscale = image.convert('L')
        ink = ImageOps.invert(grayscale)
        bbox = ink.getbbox()
        if bbox is None:
            return []
        cropped = grayscale.crop(bbox)
        boxes = self._merge_components(self._components(cropped))
        return [cropped.crop(box) for box in boxes]

    def predict_symbols(self, image, top_k=3):
        if not self.ready:
            raise RuntimeError(f'Math model unavailable: {self.error or "unknown error"}')
        symbols = self._segment(image)
        output = []
        with torch.inference_mode():
            for symbol in symbols:
                tensor = prepare_symbol_image(symbol).unsqueeze(0).to(self.device)
                probs = self.model(tensor).softmax(dim=1)[0]
                values, indexes = probs.topk(top_k)
                candidates = []
                for value, index in zip(values.tolist(), indexes.tolist()):
                    entry = self.symbols[index]
                    candidates.append((entry['latex'], value))
                output.append(candidates)
        return output

    def recognize(self, image):
        started = time.perf_counter()
        candidates = self.predict_symbols(image, top_k=3)
        tokens = [c[0][0] if c else '' for c in candidates]
        expression = ''.join(tokens)
        for source, target in self.replacements.items():
            expression = expression.replace(source, target)
        expression = expression.replace('{', '(').replace('}', ')')
        expression = re.sub(r'\\[a-zA-Z]+', '', expression)
        expression = expression.replace(' ', '')
        answer = self.evaluate(expression)
        return MathResult(expression, answer, (time.perf_counter() - started) * 1000, len(tokens))

    @staticmethod
    def evaluate(expression):
        if not expression:
            return ''
        if '=' in expression:
            try:
                import sympy as sp
                left, right = expression.split('=', 1)
                left = sp.sympify(left.replace('^', '**'))
                right = sp.sympify(right.replace('^', '**'))
                solutions = sp.solve(sp.Eq(left, right))
                return ', '.join(str(value) for value in solutions) if solutions else '0'
            except Exception:
                return '—'
        try:
            from .calculator import Calculator
            return str(Calculator.evaluate(expression))
        except Exception:
            return '—'
