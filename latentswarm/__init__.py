"""LatentSwarm: a pluggable evaluation suite for Zero-Knowledge MRTA (ZK-MRTA).

Every component is registered by name and selected via RunConfig, so the suite is extended by
writing a class and decorating it (no runner changes):

  - scenarios     latent-trait generation        (@scenario)
  - algorithms    decentralized policies          (@algorithm)
  - metrics       skill / generalization measures (@metric)
  - visualizations plots                          (@visualization)
  - env           the ZK-MRTA environment (masking, reward, offers, capacity-1 contention)

Quick start:
    from latentswarm import RunConfig
    from latentswarm.run import run
    run(RunConfig())
"""
from .config import RunConfig
from . import scenarios, algorithms, baselines, metrics   # import for side effect: populate registries
from .env import ZKMRTAEnv, reward_matrix
from .registry import SCENARIOS, ALGORITHMS, METRICS, VISUALIZATIONS

__all__ = [
    "RunConfig", "ZKMRTAEnv", "reward_matrix",
    "scenarios", "algorithms", "baselines", "metrics",
    "SCENARIOS", "ALGORITHMS", "METRICS", "VISUALIZATIONS",
]
__version__ = "0.1.0"
