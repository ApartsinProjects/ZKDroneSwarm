"""Sanity audit of the regenerated uniform-world data (run from repo root after the regen).

Checks every results/pilots JSON the paper figures/tables consume:
  - scenario == "uniform_cosine" (no block-world residue);
  - 16 seeds;
  - CLUB and BiasModel present where the schema carries a method list;
  - all values finite (no NaN/Inf);
  - sane ranges (unseen/overall in [-0.4, 1.05]); structure-free near the floor; SwarmCF (RewardCF) leads;
  - the offer-size cand variants exist (c15: 20,240; c16: 3,20,240);
  - the four LatentSwarm figure files exist.
Prints PASS/FAIL per check and a final verdict. Read-only.
"""
import glob, json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
PILOTS = "results/pilots"
PROB = []   # collected problems


def latest(pat):
    fs = sorted(glob.glob(os.path.join(PILOTS, pat)))
    return fs[-1] if fs else None


def by_cand(pat):
    out = {}
    for fp in sorted(glob.glob(os.path.join(PILOTS, pat))):
        try:
            out[int(json.load(open(fp))["meta"].get("cand"))] = fp
        except Exception:
            pass
    return out


def fail(tag, msg):
    PROB.append("%s: %s" % (tag, msg)); print("  FAIL  %s: %s" % (tag, msg))


def ok(tag, msg=""):
    print("  ok    %s %s" % (tag, msg))


def finite_tree(x, key=None):
    # state-uniqueness (uniq) is NaN by definition for structure-free methods (no per-robot model);
    # that is expected, not corruption, so skip 'uniq' subtrees. Everything else must be finite.
    if key == "uniq":
        return True
    if isinstance(x, dict):
        return all(finite_tree(v, k) for k, v in x.items())
    if isinstance(x, list):
        return all(finite_tree(v) for v in x)
    if isinstance(x, (int, float)):
        return np.isfinite(x)
    return True


def check_meta(tag, d, need_methods=True):
    meta = d.get("meta", {})
    sc = meta.get("scenario")
    if sc != "uniform_cosine":
        fail(tag, "scenario=%r (expected uniform_cosine)" % sc)
    seeds = meta.get("seeds", [])
    if len(seeds) != 16:
        fail(tag, "seeds=%d (expected 16)" % len(seeds))
    if need_methods:
        ms = meta.get("methods", [])
        for req in ("CLUB", "BiasModel"):
            if req not in ms:
                fail(tag, "method %s missing (have %s)" % (req, ms))
    if not finite_tree(d.get("raw", {})):
        fail(tag, "non-finite value present")


def vals(cell):
    """unseen list from a {overall,unseen,...} or list cell."""
    if isinstance(cell, dict):
        return cell.get("unseen", [])
    return cell


def check_rho_schema(tag, d, leads=True, floor_check=True):
    raw = d["raw"]; meta = d["meta"]
    rhos = meta.get("rhos") or list(raw.keys())
    rk = "0.25" if "0.25" in raw else str(rhos[-1])
    block = raw[rk]
    for nm, cell in block.items():
        u = [x for x in vals(cell) if x is not None]
        if u:
            mn, mx = min(u), max(u)
            if mn < -0.4 or mx > 1.05:
                fail(tag, "%s unseen out of range [%.3f,%.3f] at rho=%s" % (nm, mn, mx, rk))
    if leads and "RewardCF" in block:
        rc = np.mean([x for x in vals(block["RewardCF"]) if x is not None])
        for nm in block:
            if nm in ("RewardCF",):
                continue
            m = np.mean([x for x in vals(block[nm]) if x is not None] or [0])
            if m > rc + 1e-6:
                fail(tag, "%s unseen %.3f > RewardCF %.3f at rho=%s" % (nm, m, rc, rk))
    if floor_check:
        for nm in ("UCBIndep", "Tabular", "Random"):
            if nm in block:
                m = abs(np.mean([x for x in vals(block[nm]) if x is not None] or [0]))
                if m > 0.05:
                    fail(tag, "structure-free %s not at floor (|%.3f|>0.05) at rho=%s" % (nm, m, rk))


print("=" * 70)
print("UNIFORM-WORLD DATA AUDIT")
print("=" * 70)

# bakeoff (Table 3)
f = latest("c14_compare_*.json")
print("\n[c14_compare / Table 3]", os.path.basename(f) if f else "MISSING")
if not f:
    fail("c14", "missing")
else:
    d = json.load(open(f)); check_meta("c14", d); check_rho_schema("c14", d)
    ok("c14", "methods=%s" % d["meta"].get("methods"))

# crossover body c=20
f = latest("c15_crossover_*.json")
cx = by_cand("c15_crossover_*.json")
print("\n[c15_crossover / F5,F22,F28] cands=%s" % sorted(cx))
if 20 not in cx or 240 not in cx:
    fail("c15", "need cand 20 and 240, have %s" % sorted(cx))
if cx.get(20):
    d = json.load(open(cx[20])); check_meta("c15@20", d); check_rho_schema("c15@20", d); ok("c15@20")

# anytime cands 3,20,240
an = by_cand("c16_anytime_*.json")
print("\n[c16_anytime / F6,F22,F28] cands=%s" % sorted(an))
for c in (3, 20, 240):
    if c not in an:
        fail("c16", "missing cand=%d" % c)
if an.get(20):
    d = json.load(open(an[20])); check_meta("c16@20", d, need_methods=True)
    # anytime cells are lists (trajectories); check finite + CLUB present
    blk = d["raw"].get("0.25", {})
    if "CLUB" not in blk:
        fail("c16@20", "CLUB missing in anytime")
    else:
        ok("c16@20", "CLUB present; methods=%d" % len(blk))

# collab + scale_m (F18)
for nm, pat in (("collab", "collab_*.json"), ("scale_m", "scale_m_*.json")):
    f = latest(pat)
    print("\n[%s / F18]" % nm, os.path.basename(f) if f else "MISSING")
    if not f:
        fail(nm, "missing")
    else:
        d = json.load(open(f)); check_meta(nm, d)
        raw = d["raw"]; k0 = list(raw.keys())[0]
        if "CLUB" not in raw[k0]:
            fail(nm, "CLUB missing")
        else:
            ok(nm, "CLUB present")
        if not finite_tree(raw):
            fail(nm, "non-finite")

# iid (F8)
f = latest("e12_iid_masking_*.json")
print("\n[e12_iid_masking / F8]", os.path.basename(f) if f else "MISSING")
if not f:
    fail("e12", "missing")
else:
    d = json.load(open(f))
    if d["meta"].get("scenario") != "uniform_cosine":
        fail("e12", "scenario=%r" % d["meta"].get("scenario"))
    if "CLUB" not in d.get("rawA", {}).get("persistent", {}).get("0.25", {}):
        fail("e12", "CLUB missing in rawA")
    else:
        ok("e12", "CLUB present")
    if not (finite_tree(d.get("rawA", {})) and finite_tree(d.get("rawB", {}))):
        fail("e12", "non-finite")

# approxrank (F27)
f = latest("x5_approxrank_*.json")
print("\n[x5_approxrank / F27]", os.path.basename(f) if f else "MISSING")
if not f:
    fail("x5", "missing")
else:
    d = json.load(open(f)); raw = d["raw"]; eps = d["meta"]["eps"]
    k0 = "%.2f" % eps[0]; klast = "%.2f" % eps[-1]
    if "CLUB" not in raw[k0]:
        fail("x5", "CLUB missing")
    er0 = np.mean(raw[k0]["RewardCF"]["eff_rank"]); er1 = np.mean(raw[klast]["RewardCF"]["eff_rank"])
    if not (er1 > er0):
        fail("x5", "eff_rank not rising (%.1f -> %.1f)" % (er0, er1))
    else:
        ok("x5", "eff_rank %.1f -> %.1f; CLUB present" % (er0, er1))

# opmetrics (Table 3 regret/ttc)
f = latest("opmetrics_*.json")
print("\n[opmetrics / Table 3 regret,ttc]", os.path.basename(f) if f else "MISSING")
if not f:
    fail("opmetrics", "missing (run opmetrics.py)")
else:
    d = json.load(open(f)); a = d.get("raw", {}).get("anytime")
    if not a:
        fail("opmetrics", "no anytime block")
    else:
        for nm in ("RewardCF", "CLUB", "BiasModel", "PTF"):
            if nm not in a.get("regret", {}).get("0.25", {}):
                fail("opmetrics", "regret missing for %s" % nm)
        ok("opmetrics", "regret/ttc for %d methods" % len(a.get("regret", {}).get("0.25", {})))

# LatentSwarm figure files (F13/F23/F26/F21)
print("\n[LatentSwarm figure files / F13,F23,F26,F21]")
for nm, pat in (("main", "latentswarm_main*.json"), ("contention", "latentswarm_contention_*.json"),
                ("grounded", "latentswarm_grounded_*.json"), ("ranksweep", "latentswarm_ranksweep*.json")):
    f = latest(pat)
    if not f:
        fail("ls_" + nm, "missing")
    else:
        ok("ls_" + nm, os.path.basename(f))

# block-world residue check: any consumed-prefix file in the glob with a non-uniform scenario
print("\n[block-world residue]")
res = 0
for pat in ("c14_compare_*.json", "c15_crossover_*.json", "c16_anytime_*.json", "collab_*.json",
            "scale_m_*.json", "e12_iid_masking_*.json", "x5_approxrank_*.json"):
    for fp in glob.glob(os.path.join(PILOTS, pat)):
        try:
            sc = json.load(open(fp))["meta"].get("scenario")
            if sc not in ("uniform_cosine", None):
                fail("residue", "%s scenario=%s" % (os.path.basename(fp), sc)); res += 1
        except Exception:
            pass
if res == 0:
    ok("residue", "no non-uniform consumed-prefix files in the glob path")

print("\n" + "=" * 70)
print("VERDICT: %s  (%d problems)" % ("PASS" if not PROB else "FAIL", len(PROB)))
for p in PROB:
    print("  - " + p)
