import ast
import operator
import re


class Calculator:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression):
        expression = expression.replace('×', '*').replace('÷', '/').replace('−', '-').replace('^', '**')
        tree = ast.parse(expression, mode='eval')
        return cls._eval(tree.body)

    @classmethod
    def _eval(cls, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in cls.operators:
            return cls.operators[type(node.op)](cls._eval(node.left), cls._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls.operators:
            return cls.operators[type(node.op)](cls._eval(node.operand))
        raise ValueError('Only numeric arithmetic is supported.')

    @staticmethod
    def extract(expression):
        expression = expression.strip()
        match = re.search(r'([0-9()+\-*/×÷.^\s]+)', expression)
        return match.group(1).strip() if match else ''
