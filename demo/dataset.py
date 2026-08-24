"""BSM sample dataset: real scans from the Continual Learning Bench corpus.

Reads the ``mixed_grid_lifecycle`` corpus in place from a local clone of the
benchmark repository — the scan data is not copied into this repository.
Set ``BSM_DATA`` to the jsonl path, and ``BSM_LO``/``BSM_HI`` to slice the
90-scan schedule. The default window (scans 18-42) covers the end of stage
one and the start of stage two, so the sample includes one concept-drift
boundary.

Scoring is the benchmark's published rule: interval-set IoU between the
reported available spectrum and the true available spectrum over the 168
MHz band (the complement of the occupied intervals). Only center frequency
and bandwidth affect the score. The truth denominator here is the union of
channel definitions across the stages present in the window; the official
harness uses the union across all stages of the full schedule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BAND = (0.0, 168.0)

_DEFAULT_DATA = (Path.home() / "code" / "continual-learning-bench" / "data"
                 / "blind_spectrum_monitoring" / "mixed_grid_lifecycle.jsonl")

DATA_PATH = Path(os.environ.get("BSM_DATA") or _DEFAULT_DATA)
LO = int(os.environ.get("BSM_LO", "18"))
HI = int(os.environ.get("BSM_HI", "42"))


def load_scans() -> list[dict]:
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()
    scans = [json.loads(l) for l in lines if l.strip()]
    return scans[LO:HI]


# ---------------------------------------------------------------------------
# Interval arithmetic (the benchmark's availability-IoU rule).
# ---------------------------------------------------------------------------

def intervals_union(iv):
    iv = sorted((a, b) for a, b in iv if b > a)
    out: list[tuple[float, float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def complement(iv, band=BAND):
    iv = intervals_union(iv)
    out, cur = [], band[0]
    for a, b in iv:
        a, b = max(a, band[0]), min(b, band[1])
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < band[1]:
        out.append((cur, band[1]))
    return out


def _inter_len(A, B):
    total, i, j = 0.0, 0, 0
    A, B = intervals_union(A), intervals_union(B)
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if hi > lo:
            total += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return total


def iou_available(report_iv, truth_iv) -> float:
    """IoU between the available-spectrum sets (band complements)."""
    ra, ta = complement(report_iv), complement(truth_iv)
    inter = _inter_len(ra, ta)
    union = (sum(b - a for a, b in ra) + sum(b - a for a, b in ta) - inter)
    return inter / union if union > 0 else 1.0


def truth_intervals(scans: list[dict]) -> list[tuple[float, float]]:
    """Occupied intervals for the union of channels across the window.

    The per-scan ground-truth center frequencies carry small jitter, so
    channels are keyed by their stable id and the center is the mean.
    Wideband channels carry no bandwidth override and are 15 MHz wide;
    narrowband channels carry bandwidth_override=5.0.
    """
    acc: dict[int, list] = {}
    for r in scans:
        for g in r["ground_truth"]:
            cd = g["channel_def"]
            bw = cd.get("bandwidth_override") or 15.0
            e = acc.setdefault(cd["id"], [0.0, 0, bw])
            e[0] += g["center_freq"]
            e[1] += 1
            e[2] = max(e[2], bw)
    out = []
    for cf_sum, n, bw in acc.values():
        cf = cf_sum / n
        out.append((cf - bw / 2, cf + bw / 2))
    return sorted(out)


def peak_intervals(scan: dict) -> list[tuple[float, float]]:
    return [(p["freq_mhz"] - p["width_mhz"] / 2,
             p["freq_mhz"] + p["width_mhz"] / 2)
            for p in scan["detected_peaks"]]
