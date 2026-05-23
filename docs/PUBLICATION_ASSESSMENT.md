# Publication-fit assessment (honest), cycle 67

Honest read of where this work can publish, and what each venue needs. Material is strong (61+
catalogued result sets with CIs, a theory suite T1-T9/P6/P10-P16, fair audited baselines incl.
CBBA/SIC-MMAB/CLUB, honest negatives + sanity experiments, one unified best-or-tied method). The
question is FIT and grounding, not volume or rigor.

## Verdicts (best -> stretch)
| venue | fit | gating need |
|---|---|---|
| AAMAS / JAAMAS | BEST (pure decentralized-MAS) | essentially ready; main.tex targets AAMAS |
| Robotics & Autonomous Systems (Elsevier, IF ~4) | GOOD / REALISTIC | robotics grounding + sensing-derived observability (DONE cycle 67) + a concrete scenario; NO hardware required |
| ICRA / IROS (conf) | GOOD | one concrete grounded scenario + sim |
| IEEE T-RO / IJRR | STRETCH | real-robot / high-fidelity physics validation + close P15; abstract sim is the wrong evidence TYPE |

## IEEE T-RO (Transactions on Robotics) -- NOT ready as-is
T-RO is a robotics-SYSTEMS journal; the work is currently an ML/MAS contribution validated abstractly.
Gaps: (1) NO physical validation (synthetic block-model worlds + one PettingZoo env; T-RO expects real
robots or high-fidelity physics: dynamics, 3-D motion, real sensing, collisions); (2) thin robotics
grounding (abstract reward, no scenario/motion/energy model); (3) the keystone theory P15 is OPEN
(categorical claim cited/conjectured, not proven for persistent masking); (4) modest scale. Reaching
T-RO is a new research program (grounded sim/hardware + P15), not a polish pass.

## Robotics & Autonomous Systems (RAS) -- realistic target
RAS covers multi-robot systems, swarm robotics, decentralized coordination, MRTA, and learning for
autonomy, and ACCEPTS simulation/algorithmic + theory papers (hardware not mandatory). The MRTA +
comms-free + low-rank-CF framing, benchmarked vs CBBA-lite/SIC-MMAB/CLUB, fits squarely.
- DONE (cycle 67): sensing-grounded observability (pilot_sensing.py) -- masking+noise DERIVED from 2-D
  sensing radius + distance-noise, so rho/sigma are emergent physics; the categorical unseen win
  survives geometry-limited sensing once coverage >=~0.3 (catalogue row 62, docs/SENSING.md, Fig F16).
  Robotics grounding (p_i=robot capability profile, u_j=task requirement profile, observation=limited
  -range sensing not radio) folded into paper §2 + tutorial §8.17.
- REMAINING for a strong RAS submission: (a) one concrete SCENARIO (search/coverage/inspection) with
  capability-vs-requirement semantics in a sim; (b) feature the tabula_drone PettingZoo validation
  (catalogue row 38) as a named environment, possibly add one more; (c) reformat main.tex to Elsevier
  elsarticle class at submission. None is a new research program.

## Recommendation
Land AAMAS/JAAMAS now (material is ready). In parallel pursue RAS with the grounding track above (a few
focused cycles). Treat T-RO/IJRR as a longer follow-up gated on physical validation + P15.
