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


@scenario("sensing_coalition")
class SensingCoalition(Scenario):
    """Robotics-grounded heterogeneous sensing (STRATA / Prorok-style trait aggregation): each of the
    d latent dimensions is a physical SENSING MODALITY (e.g. EO, IR, acoustic, LiDAR, range-endurance).
    A robot's capability p_i and a task site's requirement u_j are NON-NEGATIVE profiles over those
    modalities; both are drawn as a modality archetype (a specialist, plus a small all-modality baseline)
    with jitter, so the reward R_ij=<p_i,u_j> is the modality MATCH, rank<=d, and personalized (different
    robots suit different sites). Scaled so R is O(1)."""
    name = "sensing_coalition"

    def generate(self):
        c, rng = self.cfg, self.rng
        d = c.d
        K = max(1, min(c.n_modes, d))            # modality archetypes (one specialist per modality)
        arche = 0.15 * np.ones((K, d))           # small baseline competence in every modality
        for k in range(K):
            arche[k, k % d] += 1.0               # the archetype's specialty modality
        mode_p = rng.randint(0, K, c.m); mode_u = rng.randint(0, K, c.n)
        P = np.clip(arche[mode_p] + c.jitter * rng.normal(0.0, 1.0, (c.m, d)), 0.0, None)
        U = np.clip(arche[mode_u] + c.jitter * rng.normal(0.0, 1.0, (c.n, d)), 0.0, None)
        s = float((P @ U.T).std()) + 1e-9        # scale reward to O(1)
        return P / s ** 0.5, U / s ** 0.5


def build_scenario(cfg, rng) -> Scenario:
    return get(SCENARIOS, cfg.scenario)(cfg, rng)
