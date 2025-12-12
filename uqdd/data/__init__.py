"""Data subpackage for UQDD.

This package contains dataset loaders, preparation scripts, and helper utilities
for working with Papyrus and related data sources.

Public API:
- data_papyrus: Dataset handling for Papyrus
- utils_data: Common helpers for data I/O and preprocessing
"""

# Re-export commonly used modules/functions for convenience
from . import data_papyrus as data_papyrus  # noqa: F401
from . import utils_data as utils_data  # noqa: F401

__all__ = [
    "data_papyrus",
    "utils_data",
]

