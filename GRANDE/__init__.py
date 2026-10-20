"""GRANDE: Gradient-Based Decision Tree Ensembles.

Hard, axis-aligned decision tree ensembles learned end-to-end with gradient descent.
Paper: https://openreview.net/forum?id=XEFWBxi075
"""

from ._version import __version__
from .estimator import GRANDEClassifier, GRANDERegressor

__all__ = ["GRANDEClassifier", "GRANDERegressor", "__version__"]
