"""Single-process driver: run ALL uniform-world sweeps sequentially and save to results/pilots.

ONE python process (no bash loop that respawns workers after a kill); the ProcessPoolExecutor workers
inside each sweep are children of this process and die with it. The `if __name__ == "__main__"` guard
is REQUIRED on Windows (spawn re-imports this module in every worker).

Order note: the cand=20 anytime runs LAST so the latest c16_anytime is the body (c=20) file that
opmetrics / Table 3 read; the cand=3 and cand=240 variants (for F22/F28) are written earlier and the
figure readers pick them by meta['cand'].

Run from repo root:  python experiments/run_all_sweeps.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from latentswarm import sweeps as S
    from latentswarm.sweeps import base_config, _cfg_for, _save
    out = os.path.join(ROOT, "results", "pilots")
    cfg = base_config()                       # 16 seeds, uniform_cosine

    def go(name, result):
        if isinstance(result, list):
            for p, n in result:
                print("  saved ->", _save(p, n, out))
        else:
            p, n = result
            print("  saved ->", _save(p, n, out))
        print("[R] %s ok  (%.0fs elapsed)" % (name, time.time() - t0), flush=True)

    t0 = time.time()
    print("[R] START single-process uniform regen", flush=True)
    go("bakeoff", S.sweep_bakeoff(cfg))                       # c14_compare (Table 3)
    go("offersize", S.sweep_offersize(cfg))                   # c15+c16 @ cand=20 and cand=n=240 (F5/F22/F28)
    go("anytime_c3", S.sweep_anytime(_cfg_for(cfg, offer_size=3)))  # c16 @ cand=3 (F28 in-regime)
    go("collab", S.sweep_collab(cfg))                         # F18a
    go("scale_m", S.sweep_scale_m(cfg))                       # F18b
    go("iid", S.sweep_iid_vs_persistent(cfg))                # F8
    go("approxrank", S.sweep_approxrank(cfg))                # F27
    go("anytime", S.sweep_anytime(cfg))                      # c16 @ cand=20 LAST -> latest for opmetrics/F6
    print("[R] ALL SWEEPS DONE in %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
