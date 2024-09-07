__all__ = ["DegenerateProblemError", "NoSolutionError", "OverDeterminedError"]


class DegenerateProblemError(Exception):
    """
    The problem is degenerate.

    The objective could be decreased, but only on the expense
    of violating positivity of the standard variables.
    """

    ...


class NoSolutionError(Exception):
    """
    There is no solution to this problem.

    Step size could be indefinitely increased in a
    direction without violating feasibility.
    """

    ...


class OverDeterminedError(Exception):
    """
    The problem is overdetermined.

    There are more constraints than variables, and the problem might not have "
    a feasible solution at all. Consider using a different approach."
    """

    ...
