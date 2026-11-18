"""
Compare two ION contact plans pair by pair (interval Jaccard), as in scenarios/cpd/FINDINGS.md.

    python tools/compare_contact_plans.py REF_PLAN NEW_PLAN
"""

import sys
from collections import defaultdict

import numpy as np


def read_contacts(path: str) -> dict[tuple[str, str], np.ndarray]:
    """Undirected pair -> boolean occupancy on a 600 s grid over the plan's span."""
    first = open(path).read().split("\n\n")[0]
    out = defaultdict(list)
    for line in first.strip().split("\n"):
        _, _, t0, t1, a, b, _ = line.split()
        key = tuple(sorted((a, b)))
        out[key].append((int(t0[1:]), int(t1[1:])))
    return out


def occupancy(intervals, horizon, step=600):
    occ = np.zeros(int(horizon // step), dtype=bool)
    for t0, t1 in intervals:
        occ[t0 // step : t1 // step] = True
    return occ


def category(pair):
    def c(n):
        n = int(n)
        return "relay" if n < 2000 else ("source" if n < 9000 else "gs")

    return "-".join(sorted((c(pair[0]), c(pair[1]))))


def compare(ref_path, new_path, horizon=86400):
    ref, new = read_contacts(ref_path), read_contacts(new_path)
    pairs = set(ref) | set(new)
    per_cat = defaultdict(list)
    for p in pairs:
        a = occupancy(ref.get(p, []), horizon)
        b = occupancy(new.get(p, []), horizon)
        union = (a | b).sum()
        per_cat[category(p)].append((a & b).sum() / union if union else 1.0)
    n_ref = sum(len(v) for v in ref.values())
    n_new = sum(len(v) for v in new.values())
    only_ref = len(set(ref) - set(new))
    only_new = len(set(new) - set(ref))
    allj = [j for v in per_cat.values() for j in v]
    return {
        "contacts_ref": n_ref,
        "contacts_new": n_new,
        "pairs_ref": len(ref),
        "pairs_new": len(new),
        "pairs_only_ref": only_ref,
        "pairs_only_new": only_new,
        "mean_jaccard": float(np.mean(allj)),
        **{
            f"j_{k}": round(float(np.mean(v)), 3)
            for k, v in sorted(per_cat.items())
        },
    }


if __name__ == "__main__":
    print(compare(sys.argv[1], sys.argv[2]))
