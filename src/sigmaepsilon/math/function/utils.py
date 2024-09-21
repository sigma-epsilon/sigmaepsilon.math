import re

valid_operators = ["<=", ">=", "<", ">", "="]


def parse_expression(expression: str) -> tuple[str, str, str]:
    """
    Takes an expression and returns a tuple with the left side, operator and right
    side of the expression.

    Example
    -------
    >>> from sigmaepsilon.math.function.utils import parse_expression
    >>> parse_expression("x <= 2")
    ('x', '<=', '2')

    """
    # Regular expression to match the pattern (left side, operator, right side)
    pattern = r"(.+?)\s*(<=|>=|<|>|=)\s*(.+)"

    # Use re.match to extract parts of the expression
    match = re.match(pattern, expression)

    if match:
        left_side = match.group(1).strip()
        operator = match.group(2)
        right_side = match.group(3).strip()

        for op in valid_operators:
            if op in left_side:
                raise ValueError(f"Invalid expression: '{expression}'")

            if op in right_side:
                raise ValueError(f"Invalid expression: '{expression}'")

        if operator not in valid_operators:
            raise ValueError(f"Invalid expression: '{expression}'")

        return left_side, operator, right_side
    else:
        raise ValueError(f"Invalid expression: '{expression}'")


def has_operator(expression: str) -> bool:
    """
    Returns True if the expression contains a valid operator.

    Example
    -------
    >>> from sigmaepsilon.math.function.utils import has_operator
    >>> has_operator("x <= 2")
    True

    """
    for op in valid_operators:
        if op in expression:
            return True

    return False
