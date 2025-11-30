from types import NoneType
from pydantic import BaseModel, Field
import numpy as np
from scipy.optimize import OptimizeResult


__all__ = ["OptimizerState"]


class OptimizerState(BaseModel):
    """
    Model representing the state of an optimizer.

    Attributes
    ----------
    x : list[float] | NoneType
        The current solution vector, or None if not set.
    fun : float | NoneType
        The value of the objective function at the current solution, or None if not set.
    n_fev : int
        The number of function evaluations performed.
    n_jev : int
        The number of Jacobian evaluations performed.
    n_hev : int
        The number of Hessian evaluations performed.
    n_iter : int
        The number of iterations completed.
    success : bool
        Whether or not the optimizer exited successfully.
    message : str
        A message describing the cause of the termination.
    """

    x: list[float] | NoneType = Field(
        default=None, description="The current solution vector."
    )
    fun: float | NoneType = Field(
        default=None,
        description="The value of the objective function at the current solution.",
    )
    n_fev: int = Field(
        default=0, description="The number of function evaluations performed."
    )
    n_jev: int = Field(
        default=0, description="The number of Jacobian evaluations performed."
    )
    n_hev: int = Field(
        default=0, description="The number of Hessian evaluations performed."
    )
    n_iter: int = Field(default=0, description="The number of iterations completed.")
    success: bool = Field(
        default=True, description="Indicates if the optimizer exited successfully."
    )
    message: str = Field(
        default="", description="Description of the cause of termination."
    )
    stage: int | NoneType = Field(
        default=None,
        description=(
            "Termination status of the optimizer. "
            "Its value depends on the underlying solver. "
            "Refer to the solver being used for more details."
        ),
    )

    def to_scipy(self) -> OptimizeResult:
        """Returns the state as a SciPy OptimizeResult object."""
        return OptimizeResult(
            x=np.array(self.x) if self.x is not None else None,
            fun=self.fun,
            status=self.stage,
            nfev=self.n_fev,
            njev=self.n_jev,
            nhev=self.n_hev,
            success=self.success,
            message=self.message,
            nit=self.n_iter,
        )
