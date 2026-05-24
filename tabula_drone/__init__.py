"""
TabulaDrone: Reinforcement Learning Environments for Drone Target Engagement.

DEPRECATED. This package is superseded by the `latentswarm` package, the ground-up,
modular, pluggable ZK-MRTA evaluation suite used by the paper (see latentswarm/ and
experiments/run via `python -m latentswarm.run`). tabula_drone is retained only for
historical reference and is no longer maintained; new work should use latentswarm.
"""
import warnings

__version__ = "0.1.0"

warnings.warn(
    "tabula_drone is deprecated and superseded by the 'latentswarm' package; "
    "use latentswarm for new work.",
    DeprecationWarning,
    stacklevel=2,
)
