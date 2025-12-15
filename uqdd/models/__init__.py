"""Models subpackage for UQDD

The ``uqdd.models`` subpackage hosts model architectures, training utilities,
losses, and parsers for uncertainty-aware drug discovery. It provides
implementations for evidential learning, MC Dropout, ensembles, and PNN, plus
helpers for configuration, metrics during training, and training loops.

Modules
-------
- ``ensemble``: EnsembleDNN and helpers for bagging/aggregation and inference.
- ``evidential``: Evidential deep learning utilities, priors, and losses.
- ``mcdropout``: Monte Carlo Dropout training and inference routines.
- ``pnn``: Base probabilistic neural network model and components.
- ``emc``: Error-model calibration utilities.
- ``eoe``: Evidential-on-ensembles routines to combine EDL with ensembles.
- ``loss``: Loss functions used across model families.
- ``model_parser``: CLI/arg parsing for models and experiments.
- ``utils_metrics``: Metrics utilities for training/evaluation (loss/metrics hooks).
- ``utils_models``: Model configuration and helper utilities.
- ``utils_train``: Training loops, dataloaders, evaluation pipelines.

Lazy Loading
------------
Submodules are loaded lazily on first attribute access to minimize import-time
side effects, e.g., ``uqdd.models.ensemble`` triggers loading only when used.
This keeps CLI startup fast and reduces dependencies during basic imports.

Usage Notes
-----------
- Configuration: Use JSON files from ``uqdd/config`` and the ``model_parser``
  helper to create reproducible experiments.
- Device: Respect the global ``DEVICE`` from ``uqdd.__init__`` when constructing
  models and dataloaders.
- Logging: Prefer logging over printing during training; write under
  ``uqdd/logs`` and use callbacks for metrics collection.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ensemble, evidential, mcdropout, pnn, emc, eoe, loss, model_parser, utils_metrics, utils_models, utils_train

__all__ = [
    "ensemble",
    "evidential",
    "mcdropout",
    "pnn",
    "emc",
    "eoe",
    "loss",
    "model_parser",
    "utils_metrics",
    "utils_models",
    "utils_train",
]

_submodules = {
    "ensemble": ".ensemble",
    "evidential": ".evidential",
    "mcdropout": ".mcdropout",
    "pnn": ".pnn",
    "emc": ".emc",
    "eoe": ".eoe",
    "loss": ".loss",
    "model_parser": ".model_parser",
    "utils_metrics": ".utils_metrics",
    "utils_models": ".utils_models",
    "utils_train": ".utils_train",
}


def __getattr__(name):
    try:
        mod_path = _submodules[name]
    except KeyError as e:
        raise AttributeError(f"module 'uqdd.models' has no attribute {name!r}") from e
    from importlib import import_module
    return import_module(mod_path, package=__name__)


def __dir__():
    return sorted(list(__all__))
