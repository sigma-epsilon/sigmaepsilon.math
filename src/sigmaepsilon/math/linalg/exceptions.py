from sigmaepsilon.core.exceptions import SigmaEpsilonException


class TensorShapeMismatchError(SigmaEpsilonException):
    """Mismatch in the shape of the inputs."""


class LinalgOperationInputError(SigmaEpsilonException):
    """Invalid input for this operation."""


class LinalgMissingInputError(SigmaEpsilonException):
    """Invalid input for this operation."""


class LinalgInvalidTensorOperationError(SigmaEpsilonException):
    """
    Tensors don't support this operation. Try to call this using
    the arrays of the tensorial inputs.
    """


class LinalgError(SigmaEpsilonException):
    """General linear algebra error"""
