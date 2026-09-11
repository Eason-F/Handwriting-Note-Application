from __future__ import annotations

import re
from dataclasses import dataclass

import sympy
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations


class MathEvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class MathResult:
    expression: str
    result: str
    assignment: str | None = None


class MathEvaluator:
    """A per-document symbolic context without Python eval or arbitrary functions."""

    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
    FUNCTIONS = {
        'sqrt': sympy.sqrt, 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'log': sympy.log, 'exp': sympy.exp, 'abs': sympy.Abs,
        'expand': sympy.expand, 'factor': sympy.factor, 'simplify': sympy.simplify,
        'solve': sympy.solve,
    }
    CONSTANTS = {'pi': sympy.pi, 'E': sympy.E, 'I': sympy.I}

    def __init__(self):
        self.variables: dict[str, sympy.Basic] = {}

    @staticmethod
    def normalize(expression: str):
        return expression.strip().replace('×', '*').replace('÷', '/').replace('−', '-').replace('–', '-').replace('π', 'pi')

    def evaluate(self, expression: str):
        expression = self.normalize(expression)
        if not expression:
            raise MathEvaluationError('Select an expression or place the cursor on a non-empty line.')
        query = re.fullmatch(r'(.+?)\s*=\s*(.+?)\s+([A-Za-z]\w*)\s*=\s*\?', expression, re.DOTALL)
        if query:
            left, right, name = query.groups()
            symbol = sympy.Symbol(name)
            solutions = sympy.solve(sympy.Eq(self._parse(left, {name: symbol}), self._parse(right, {name: symbol})), symbol)
            if not solutions:
                raise MathEvaluationError(f'No solution found for {name}.')
            formatted = ', '.join(self.format_result(value) for value in solutions)
            return MathResult(expression, f'{name} = {formatted}')
        assignment = re.fullmatch(r'([A-Za-z]\w*)\s*=\s*(.+)', expression, re.DOTALL)
        if assignment:
            return self.assign(assignment.group(1), assignment.group(2), expression)
        value = self._parse(expression)
        return MathResult(expression, self.format_result(value))

    def assign(self, name: str, expression: str, original: str | None = None):
        if name in self.FUNCTIONS or name in self.CONSTANTS or name.startswith('_'):
            raise MathEvaluationError(f'“{name}” cannot be used as a variable name.')
        value = sympy.simplify(self._parse(expression))
        self.variables[name] = value
        return MathResult(original or f'{name} = {expression}', self.format_result(value), name)

    def _parse(self, expression: str, symbol_overrides=None):
        if '__' in expression or re.search(r'''[\[\]{};:"'`]|\b(?:import|lambda|exec|eval|open)\b''', expression):
            raise MathEvaluationError('That expression contains unsupported syntax.')
        identifiers = set(re.findall(r'\b[A-Za-z]\w*\b', expression))
        unknown_calls = {name for name in re.findall(r'\b([A-Za-z]\w*)\s*\(', expression) if name not in self.FUNCTIONS}
        if unknown_calls:
            raise MathEvaluationError(f'Unknown function: {sorted(unknown_calls)[0]}')
        local = dict(self.CONSTANTS)
        local.update(self.FUNCTIONS)
        local.update(self.variables)
        local.update(symbol_overrides or {})
        for name in identifiers - local.keys():
            local[name] = sympy.Symbol(name)
        try:
            value = parse_expr(expression, local_dict=local, transformations=self.TRANSFORMATIONS, evaluate=True)
        except Exception as exc:
            raise MathEvaluationError(f'Invalid mathematical expression: {exc}') from exc
        if value is sympy.zoo or value is sympy.nan or getattr(value, 'has', lambda *_: False)(sympy.zoo, sympy.nan):
            raise MathEvaluationError('The expression is undefined (possibly division by zero).')
        return value

    def clear_variables(self):
        self.variables.clear()

    @staticmethod
    def format_result(value):
        if isinstance(value, (list, tuple, set)):
            return '[' + ', '.join(sympy.sstr(item) for item in value) + ']'
        return sympy.sstr(value)

    @staticmethod
    def extract(text: str):
        return text.strip()


class Calculator:
    """Compatibility facade for callers that only need a one-off result."""

    @classmethod
    def evaluate(cls, expression):
        result = MathEvaluator().evaluate(expression).result
        try:
            return int(result)
        except ValueError:
            try:
                return float(result)
            except ValueError:
                return result

    extract = staticmethod(MathEvaluator.extract)
