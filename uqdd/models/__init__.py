"""UQDD models package.

Submodules:
- ensemble: EnsembleDNN and helpers
- evidential: Evidential DL utilities and losses
- mcdropout: Monte Carlo Dropout training and inference
- pnn: Base probabilistic neural network model
- emc: Error-model calibration utilities
- eoe: Evidential-on-ensembles routines
- loss: Loss functions
- model_parser: CLI/arg parsing for models
- utils_metrics: Metrics utilities for training/eval
- utils_models: Model config and helpers
- utils_train: Training loops, dataloaders, evaluation

Access submodules as attributes, e.g., `uqdd.models.ensemble`. They are
loaded lazily on first access to minimize import-time side effects.
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
