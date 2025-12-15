"""Data subpackage for UQDD

The ``uqdd.data`` subpackage provides dataset loaders, preparation scripts,
and helper utilities for reproducible data workflows with Papyrus and related
sources. It centralizes data I/O, preprocessing, and dataset management.

Modules
-------
- ``data_papyrus``: Dataset handling utilities tailored to the Papyrus dataset
    (loading, filtering, and standardization), plus example preparation scripts.
- ``utils_data``: Common helpers for data I/O, preprocessing, splitting, and
    misc utilities used by datasets and training.

Public API
----------
The most commonly used entry points are re-exported for convenience:
- ``data_papyrus``
- ``utils_data``

Usage Notes
-----------
- Configuration: Refer to ``uqdd/config`` for dataset-specific settings.
- Reproducibility: Prefer functions that take seeds and emit logs under
    ``uqdd/logs`` to ensure traceability of data preparation.
- Paths: Use the global paths defined in ``uqdd.__init__`` (e.g., DATA_DIR,
    DATASET_DIR) to keep file operations consistent across the project.
"""

# Re-export commonly used modules/functions for convenience
from . import data_papyrus as data_papyrus  # noqa: F401
from . import utils_data as utils_data  # noqa: F401

__all__ = [
    "data_papyrus",
    "utils_data",
]
