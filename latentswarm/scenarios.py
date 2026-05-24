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
        arche = c.sensing_base_competence * np.ones((K, d))   # small baseline competence in every modality
        for k in range(K):
            arche[k, k % d] += c.sensing_specialty           # the archetype's specialty modality
        mode_p = rng.randint(0, K, c.m); mode_u = rng.randint(0, K, c.n)
        P = np.clip(arche[mode_p] + c.jitter * rng.normal(0.0, 1.0, (c.m, d)), 0.0, None)
        U = np.clip(arche[mode_u] + c.jitter * rng.normal(0.0, 1.0, (c.n, d)), 0.0, None)
        s = float((P @ U.T).std()) + 1e-9        # scale reward to O(1)
        return P / s ** 0.5, U / s ** 0.5


@scenario("block_cosine")
class BlockCosine(Scenario):
    """PARITY scenario: a faithful port of experiments/core.make_world for signed=True (the
    analytical harness's UNIT-COSINE world). K1=K2=n_types latent types; type prototypes and
    per-entity traits are L2-normalized so R = P U^T is a signed COSINE in [-1, 1] (rank <= min(d,
    n_types)); every latent type is forced present (first K rows take ids 0..K-1, then shuffled),
    with within-type jitter = cfg.jitter (core's `within`, default 0.15 there).

    RNG parity: this reproduces make_world's exact RandomState(seed) draw order
    (A, B, t, s, shuffle t, shuffle s, P-jitter, U-jitter), so given the runner's world_rng =
    RandomState(seed) the produced P, U are bit-identical to core.make_world(..., signed=True).
    Use this (not gaussian_mixture, which is the UNNORMALIZED inner-product world) to match the
    analytical numbers."""
    name = "block_cosine"

    @staticmethod
    def _unit(X):
        # signed (no abs): unit-L2-normalize rows -> inner product becomes cosine in [-1, 1].
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def generate(self):
        c, rng = self.cfg, self.rng
        m, n, d = c.m, c.n, c.d
        K1 = K2 = int(c.n_types)
        within = c.jitter
        A = self._unit(rng.randn(K1, d))        # drone-type prototypes (signed, unit)
        B = self._unit(rng.randn(K2, d))        # target-type prototypes (signed, unit)
        t = rng.randint(0, K1, m); s = rng.randint(0, K2, n)
        t[:K1] = np.arange(K1); s[:K2] = np.arange(K2)   # force every latent type present
        rng.shuffle(t); rng.shuffle(s)
        P = self._unit(A[t] + within * rng.randn(m, d))
        U = self._unit(B[s] + within * rng.randn(n, d))
        return P, U


@scenario("uniform_cosine")
class UniformCosine(Scenario):
    """Uniform-on-the-sphere traits: i.i.d. Gaussian, L2-normalized so R = P U^T is a signed
    COSINE in [-1, 1] (rank <= d), with NO discrete types (every robot/task individually distinct).
    The no-clusters analog of block_cosine, matching experiments/core.make_world(model="uniform",
    signed=True). Harder for unseen-pair recovery than the block world (no type to generalize from),
    so it is the honest stress on whether continuous low-rank structure alone carries the method."""
    name = "uniform_cosine"

    @staticmethod
    def _unit(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def generate(self):
        c, rng = self.cfg, self.rng
        P = self._unit(rng.randn(c.m, c.d))         # i.i.d. on the unit sphere (no types)
        U = self._unit(rng.randn(c.n, c.d))
        return P, U


@scenario("approx_lowrank")
class ApproxLowRank(Scenario):
    """Approximately low-rank reward (Appendix F robustness): a clean rank-d unit-sphere reward
    R0 = P0 U0^T perturbed by a FULL-RANK Gaussian term, R_eps = (R0 + eps*s*G)/sqrt(1+eps^2) with
    s = std(R0)/std(G), so the entry-wise scale is fixed while energy leaves the rank-d subspace (the
    low-rank energy fraction is 1/(1+eps^2); effective rank rises from d as eps=cfg.approx_eps grows).
    Returned in FACTORED form (wide factors) so the env reward P U^T and the held-out ground truth are
    both EXACTLY R_eps. eps=0 reduces to the clean uniform-sphere rank-d world."""
    name = "approx_lowrank"

    @staticmethod
    def _unit(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def generate(self):
        c, rng = self.cfg, self.rng
        m, n, d = c.m, c.n, c.d
        eps = float(getattr(c, "approx_eps", 0.0))
        P0 = self._unit(rng.randn(m, d)); U0 = self._unit(rng.randn(n, d))
        if eps <= 0:
            return P0, U0
        r = min(m, n)                                   # full-rank perturbation (rank min(m,n))
        Gp = rng.randn(m, r); Gu = rng.randn(n, r)
        s = float((P0 @ U0.T).std() / ((Gp @ Gu.T).std() + 1e-12))
        a = 1.0 / np.sqrt(1.0 + eps * eps); b = eps * s * a
        P = np.hstack([a * P0, b * Gp])                 # (m, d+r): P U^T = (R0 + eps s G)/sqrt(1+eps^2)
        U = np.hstack([U0, Gu])                         # (n, d+r)
        return P, U


def build_scenario(cfg, rng) -> Scenario:
    return get(SCENARIOS, cfg.scenario)(cfg, rng)
