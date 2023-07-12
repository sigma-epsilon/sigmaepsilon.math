import importlib.metadata

from .hist import *
from .utils import *
from .decorate import *

__pkg_name__ = "sigmaepsilon.math"
__version__ = importlib.metadata.version(__pkg_name__)
__description__ = "A Python Library for Applied Mathematics in Physical Sciences."
