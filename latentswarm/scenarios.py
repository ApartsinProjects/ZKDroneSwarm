"""Pluggable scenario builders: generate the hidden low-rank latent traits P (robots) and
U (tasks) so that R = P U^T is the rank-d reward. Register new generators with @scenario(name)."""
import numpy as np

from .registry import scenario, get, SCENARIOS


class Scenario:
    """Base class. Subclasses implement generate() -> (P [m,d], U [n,d])."""
    name = "base"

    def __init__(self, cfg, rng: np.random.RandomState):
        self.cfg = cfg
        self.rng = rng

    def generate(self):
        raise NotImplementedError


@scenario("gaussian_mixture")
class GaussianMixture(Scenario):
    """Signed Gaussian mixture (block model): robots and tasks each belong to one of n_modes
    latent types, sharing a signed type center plus small jitter. Observing a few tasks of a
    type recovers that type's factor and generalizes to its unseen tasks. Traits are scaled so
    R_ij = <p_i, u_j> is O(1)."""
    name = "gaussian_mixture"

    def generate(self):
        c, rng = self.cfg, self.rng
        centers = rng.normal(0.0, 1.0, (c.n_modes, c.d))
        mode_p = rng.randint(0, c.n_modes, c.m)
        mode_u = rng.randint(0, c.n_modes, c.n)
        P = (centers[mode_p] + c.jitter * rng.normal(0.0, 1.0, (c.m, c.d))) / (c.d ** 0.25)
        U = (centers[mode_u] + c.jitter * rng.normal(0.0, 1.0, (c.n, c.d))) / (c.d ** 0.25)
        return P, U


@scenario("iid_gaussian")
class IIDGaussian(Scenario):
    """Signed i.i.d. Gaussian traits (every robot/task distinct; no block structure). Harder
    for unseen-pair recovery because each task must be individually observed enough times."""
    name = "iid_gaussian"

    def generate(self):
        c, rng = self.cfg, self.rng
        P = rng.normal(0.0, 1.0, (c.m, c.d)) / (c.d ** 0.25)
        U = rng.normal(0.0, 1.0, (c.n, c.d)) / (c.d ** 0.25)
        return P, U


def build_scenario(cfg, rng) -> Scenario:
    return get(SCENARIOS, cfg.scenario)(cfg, rng)
