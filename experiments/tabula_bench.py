"""
LatentSwarm faithful ZK-MRTA validation (Section 6.5 transfer test).

A SECOND, independently-implemented instantiation of the Section 3 setting, built on the
tabula_drone PettingZoo/Gymnasium environment, that mirrors the analytical harness
faithfully and ADDS capacity-1 contention:

  - reward    : signed low-rank inner product R_ij = <p_i, u_j> (reward_mode="dot"),
                signed zero-mean Gaussian traits (rank d), no all-positive popularity tilt
  - observe   : persistent Bernoulli(rho) partial-visibility mask between robots PLUS
                independent per-observer (private) reward noise; no two robots see the
                same stream (the heart of the setting)
  - scarcity  : n >> T (each robot engages O(T) of the n tasks in one mission)
  - rank      : every structured method uses the SAME guessed rank d-hat (fair)
  - contention: capacity-1 (only the first robot to pick a target each round succeeds)
  - NO 2-D positions enter the reward, the dynamics, or any policy (dropped; use a
                t-SNE/PCA of the latent traits only for visualization)

Two metrics:
  - unseen-pair skill (the categorical headline): each robot's decision quality on tasks
    it NEVER engaged, using its learned model; self-normalized so oracle=1, random~0.
    Independent of within-round contention.
  - earned (anytime) skill: mean reward over the mission, normalized to random/oracle,
    where the oracle is the per-round one-to-one (Hungarian) capacity-1 matching ceiling.
    The gap reflects estimation AND un-de-conflicted contention.

Run from the repo root:  python experiments/tabula_bench.py
"""
import os, sys, json
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.ucb_indep_policy import UCBIndepPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.weighted_als_policy import WeightedALSPolicy

# --- ZK-MRTA faithful configuration (mirrors the Section 3 analytical harness) ---
M = 30            # robots
N = 240           # tasks (n >> T: task-scarce)
D = 5             # true latent rank
DHAT = 8          # guessed rank, SHARED by every structured method (fair comparison)
KMODES = 5        # latent task/robot types (block/mixture model, as in the analytical harness)
C = 20            # per-robot offered subset size each round (the size-c offer of Section 3)
T = 50            # mission horizon (rounds); n >> T (each robot engages <= T of n tasks)
RHO = 0.5         # persistent partial-broadcast rate
SIGMA_OBS = 0.3   # per-observer (private) reward noise on the broadcast
SEEDS = list(range(16))
PTYPES = ["random", "oracle", "ucb_indep", "mf", "weighted_als"]
RUN_IN_ENV = ["random", "ucb_indep", "mf", "weighted_als"]   # oracle is analytic


def make_traits(seed):
    """Signed Gaussian-mixture traits (block model, as in the analytical harness): robots
    and tasks each belong to one of KMODES latent types with a shared signed center plus
    small jitter, so observing a few tasks of a type recovers that type's factor and
    generalizes to its unseen tasks. R_ij = <p_i,u_j> is signed, low-rank, O(1)."""
    rng = np.random.RandomState(seed)
    centers = rng.normal(0.0, 1.0, (KMODES, D))
    mp = rng.randint(0, KMODES, M)
    mu = rng.randint(0, KMODES, N)
    P = (centers[mp] + 0.2 * rng.normal(0.0, 1.0, (M, D))) / (D ** 0.25)
    U = (centers[mu] + 0.2 * rng.normal(0.0, 1.0, (N, D))) / (D ** 0.25)
    return P, U


def build_env(P, U):
    dcfg = [{"position": (0.0, 0.0), "mode_id": 0,
             "latent_vector": tuple(float(v) for v in P[i])} for i in range(M)]
    tcfg = [{"position": (0.0, 0.0), "mode_id": 0,
             "latent_vector": tuple(float(v) for v in U[j])} for j in range(N)]
    return DroneEngageLatentMRTA(
        world_size=(1.0, 1.0), max_steps=T, drones_config=dcfg, targets_config=tcfg,
        scenario_id="zk_faithful", reward_noise=SIGMA_OBS, observation_noise=0.0,
        effect_noise=0.0, target_hp=1e9, broadcast_rho=RHO, reward_mode="dot",
        capacity_one=True, offer_size=C, latent_world={"mode": "signed", "latent_dim": D})


def make_policy(ptype, seed):
    if ptype == "random":
        return RandomPolicy(seed=seed, allow_noop=False)
    if ptype == "ucb_indep":
        return UCBIndepPolicy(num_agents=M, num_targets=N, c=2.0, seed=seed, allow_noop=False)
    # Both structured methods use the SAME guessed rank d-hat and the SAME exploration
    # schedule as the Section 3 analytical harness (eps0=0.5, decay 0.93, min 0.05).
    # Both structured methods use the SAME guessed rank d-hat and the SAME exploration
    # schedule (eps0=0.4, decay 0.99, min 0.05): enough exploration for broad coverage so
    # task factors are recovered for unseen-pair prediction.
    if ptype == "mf":
        pol = {f"drone_{i}": MatrixFactorizationPolicy(
                   num_targets=N, agent_idx=i, num_agents=M, latent_dim=DHAT,
                   learning_rate=0.05, lambda_reg=1e-2, epsilon=0.4, epsilon_decay=0.99,
                   epsilon_min=0.05, anti_signal_weight=1.0, use_integration_matrix=False,
                   seed=seed + i) for i in range(M)}
        return MultiAgentPolicy(pol)
    if ptype == "weighted_als":
        pol = {f"drone_{i}": WeightedALSPolicy(
                   num_targets=N, agent_idx=i, num_agents=M, latent_dim=DHAT, lam=1.0,
                   als_sweeps=8, refit_every=3, epsilon=0.4, epsilon_decay=0.99,
                   epsilon_min=0.05, seed=seed + i) for i in range(M)}
        return MultiAgentPolicy(pol)
    raise ValueError(ptype)


def oracle_per_step(P, U):
    """Capacity-1 ceiling: optimal one-to-one (Hungarian) matching of m robots to m
    distinct tasks maximizing total <p_i,u_j>; same every round (static traits)."""
    R = P @ U.T
    ri, ci = linear_sum_assignment(-R)          # maximize
    return float(R[ri, ci].sum() / M)


def run_mission(env, policy, seed):
    """One T-round mission; returns (per-round mean reward, per-robot engaged-target sets)."""
    obs, infos = env.reset(seed=seed)
    ref = env.agents[0]; per_round = []; done = False
    engaged = [set() for _ in range(M)]
    while not done:
        try:
            actions = policy.select_actions(obs, infos, env=env)
        except TypeError:
            actions = policy.select_actions(obs, infos)
        for aid, a in actions.items():
            if a and int(a) > 0:
                engaged[int(aid.split("_")[1])].add(int(a) - 1)
        obs, rewards, term, trunc, infos = env.step(actions)
        try:
            policy.update(obs)
        except Exception:
            pass
        per_round.append(float(np.mean(list(rewards.values()))))
        done = bool(term[ref]) or bool(trunc[ref])
    return per_round, engaged


def pred_rows(policy):
    """(M x N) predicted-reward matrix if the policy holds a low-rank model, else None."""
    try:
        st = policy.get_learning_state()
    except Exception:
        return None
    if not st or "agents" not in st:
        return None
    rows = []
    for k, a in enumerate(st["agents"]):
        if not a or "P" not in a or "U" not in a:
            return None
        Ph = np.asarray(a["P"], float); Uh = np.asarray(a["U"], float)
        rows.append(Ph[a.get("agent_idx", k)] @ Uh)
    return np.asarray(rows)


def unseen_skill(P, U, pred, engaged, rng):
    """Decision quality on NEVER-engaged tasks; self-normalized (oracle=1, random~0)."""
    R = P @ U.T
    sks = []
    for i in range(M):
        unseen = np.array([j for j in range(N) if j not in engaged[i]])
        if unseen.size < 2:
            continue
        r = R[i, unseen]
        denom = r.max() - r.mean()
        if denom < 1e-9:
            continue
        if pred is not None:
            jrel = int(np.argmax(pred[i, unseen]))
        else:
            jrel = int(rng.randint(unseen.size))   # no model -> random pick (structure-free floor)
        sks.append(float((R[i, unseen[jrel]] - r.mean()) / denom))
    return float(np.mean(sks)) if sks else None


def _ci(xs, B=5000, seed=0):
    a = np.asarray([x for x in xs if x is not None], float)
    if a.size == 0:
        return (None, None, None)
    if a.size == 1:
        return (float(a[0]), float(a[0]), float(a[0]))
    rng = np.random.RandomState(seed)
    boot = a[rng.randint(0, a.size, size=(B, a.size))].mean(1)
    return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    raw = {p: [] for p in PTYPES}      # per-seed mean reward over the mission
    traj = {p: [] for p in PTYPES}     # per-seed per-round reward (learning curve)
    uraw = {p: [] for p in PTYPES}     # per-seed unseen-pair skill (self-normalized)
    for s in SEEDS:
        P, U = make_traits(s)
        ostep = oracle_per_step(P, U)
        raw["oracle"].append(ostep); traj["oracle"].append([ostep] * T); uraw["oracle"].append(1.0)
        rng = np.random.RandomState(10000 + s)
        print("seed %d: oracle/step=%.4f" % (s, ostep))
        for p in RUN_IN_ENV:
            env = build_env(P, U)
            pol = make_policy(p, seed=1000 + s)
            pr, engaged = run_mission(env, pol, seed=s)
            traj[p].append(pr); raw[p].append(float(np.mean(pr)))
            uraw[p].append(unseen_skill(P, U, pred_rows(pol), engaged, rng))
            print("  %-13s earned=%7.4f  unseen=%6.3f" % (p, raw[p][-1], uraw[p][-1]))
    rnd = np.array(raw["random"]); orc = np.array(raw["oracle"])
    eskills = {}
    print("\n%-13s | earned skill        | unseen-pair skill" % "policy")
    print("-" * 64)
    for p in PTYPES:
        v = np.array(raw[p]); sk = (v - rnd) / np.maximum(orc - rnd, 1e-9)
        eskills[p] = sk.tolist()
        em, elo, ehi = _ci(sk.tolist()); um, ulo, uhi = _ci(uraw[p])
        us = "%.3f [%.3f, %.3f]" % (um, ulo, uhi) if um is not None else "n/a"
        print("%-13s | %.3f [%.3f, %.3f] | %s" % (p, em, elo, ehi, us))
    print("-" * 64)
    os.makedirs("results/pilots", exist_ok=True)
    out = "results/pilots/tabula_bench_real.json"
    json.dump({"meta": {"experiment": "LatentSwarm faithful ZK-MRTA validation",
                        "m": M, "n": N, "d": D, "dhat": DHAT, "T": T, "rho": RHO,
                        "sigma_obs": SIGMA_OBS, "capacity_one": True,
                        "reward": "signed inner product", "oracle": "Hungarian capacity-1 matching",
                        "seeds": SEEDS, "policies": PTYPES,
                        "metric": "unseen-pair skill (categorical) + earned (anytime) skill"},
               "raw_reward": raw, "skill": eskills, "uskill": uraw, "traj": traj},
              open(out, "w"), indent=0)
    print("saved ->", out)


if __name__ == "__main__":
    main()
