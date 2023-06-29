class TensorShapeMismatchError(Exception):
    """Mismatch in the shape of the inputs."""


class LinalgOperationInputError(Exception):
    """Invalid input for this operation."""


class LinalgMissingInputError(Exception):
    """Invalid input for this operation."""


class LinalgInvalidTensorOperationError(Exception):
    """
    Tensors don't support this operation. Try to call this using
    the arrays of the tensorial inputs.
    """
