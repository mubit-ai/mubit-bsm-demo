"""BSM sample scenario: three retention strategies over real benchmark scans.

Each scan runs through three arms in lockstep. Every arm makes one LLM
call per scan with the same base prompt and the same current-scan peaks.

- off: holds no state of any kind between scans.
- icl (context replay): holds no external memory, but resends every
  earlier scan verbatim in the prompt. Its prompt at scan N contains the
  N-1 earlier scan blocks plus the current one.
- on: holds a transmitter registry that lives in Mubit. It recalls the
  registry before the call, merges the current peaks into it, and writes
  the update back after the call. The registry block in the prompt has a
  bounded size (one line per known transmitter).

All reports are scored with the benchmark's availability-IoU rule
against the window's latent channel set.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

from mubit import Client

from demo.llm_stream import complete
from demo.dataset import (
    load_scans, truth_intervals, peak_intervals, iou_available, BAND, LO, HI,
)

_ENDPOINT = os.environ.get("MUBIT_ENDPOINT", "http://127.0.0.1:3320")
_API_KEY = os.environ.get("MUBIT_API_KEY", "")

_REG_TAG = "[bsm-registry]"

# The task statement, identical for BOTH arms. It does not describe the
# band layout: neither arm gets structure it has not observed.
_BASE_SYSTEM = (
    "You are a spectrum monitoring analyst. You receive RF scan data from a "
    "168 MHz band and must report ALL persistent transmitters in the band. "
    "Transmitters are often temporarily inactive: a transmitter that was "
    "observed at any earlier point usually still exists even when it is "
    "not visible in the current scan. Report every transmitter you have "
    "evidence for, with its center frequency and bandwidth in MHz.\n"
    "Respond with ONLY a JSON object, no prose:\n"
    '{"transmitters": [{"center_freq": <MHz>, "bandwidth": <MHz>}]}'
)


def _fmt_peaks(scan: dict) -> str:
    if not scan["detected_peaks"]:
        return "(no peaks above the noise floor)"
    return "\n".join(
        f"  freq: {p['freq_mhz']:.1f} MHz | power: {p['power_dbm']:.1f} dBm "
        f"| width: {p['width_mhz']:.1f} MHz"
        for p in scan["detected_peaks"])


def _fmt_registry(reg: list[dict]) -> str:
    if not reg:
        return "(no transmitters accumulated yet)"
    return "\n".join(
        f"  {e['center_freq']:>7.1f} MHz | bw={e['bandwidth']:>5.1f} | "
        f"hits={e['hit_count']:>2} | {'confirmed' if e['hit_count'] >= 2 else 'tentative'}"
        for e in reg)


def _merge(reg: list[dict], scan: dict, scan_num: int) -> list[str]:
    """Merge the scan's peaks into the registry. Returns new-entry labels."""
    added = []
    for p in scan["detected_peaks"]:
        f, w = p["freq_mhz"], p["width_mhz"]
        wide = w >= 10.0
        best, best_d = -1, 1e9
        for i, e in enumerate(reg):
            if (e["bandwidth"] >= 10.0) != wide:
                continue
            d = abs(e["center_freq"] - f)
            if d < best_d:
                best_d, best = d, i
        if best >= 0 and best_d < 8.0:
            e = reg[best]
            n = e["hit_count"] + 1
            e["center_freq"] = round((e["center_freq"] * e["hit_count"] + f) / n, 2)
            e["bandwidth"] = round((e["bandwidth"] * e["hit_count"] + w) / n, 1)
            e["hit_count"] = n
            e["last_seen"] = scan_num
        else:
            reg.append({"center_freq": round(f, 2), "bandwidth": round(w, 1),
                        "hit_count": 1, "first_seen": scan_num,
                        "last_seen": scan_num})
            added.append(f"{f:.1f} MHz")
    reg.sort(key=lambda e: e["center_freq"])
    return added


def _parse_report(text: str) -> tuple[list[tuple[float, float]], int, bool]:
    """Parse the LLM's JSON report into occupied intervals."""
    try:
        lo, hi = text.find("{"), text.rfind("}")
        obj = json.loads(text[lo:hi + 1])
        iv = []
        for t in obj.get("transmitters", []):
            cf, bw = float(t["center_freq"]), float(t["bandwidth"])
            iv.append((cf - bw / 2, cf + bw / 2))
        return iv, len(iv), True
    except Exception:
        pass
    # Fallback: per-object field pairs anywhere in the text, either order.
    iv = []
    for obj in re.findall(r"\{[^{}]*\}", text):
        mc = re.search(r'"center_freq":\s*(-?[\d.]+)', obj)
        mb = re.search(r'"bandwidth":\s*(-?[\d.]+)', obj)
        if mc and mb:
            cf, bw = float(mc.group(1)), float(mb.group(1))
            iv.append((cf - bw / 2, cf + bw / 2))
    return iv, len(iv), False


class BSMScenario:
    name = "bsm"
    label = "CL-Bench BSM sample"

    def run(self, emit) -> None:
        scans = load_scans()
        truth = truth_intervals(scans)
        run_id = f"bsm-demo-{uuid.uuid4().hex[:8]}"

        mubit = Client(endpoint=_ENDPOINT, api_key=_API_KEY, transport="http")
        mubit.set_run_id(run_id)

        emit({"type": "meta", "run_id": run_id, "scans": len(scans),
              "window": [LO, HI], "band": list(BAND),
              "truth": [{"cf": round((a + b) / 2, 2), "bw": round(b - a, 1)}
                        for a, b in truth]})

        scores = {"on": [], "off": [], "icl": []}
        registry: list[dict] = []
        replay_history: list[str] = []
        last_stage: Optional[int] = None

        for i, scan in enumerate(scans, 1):
            stage = scan.get("stage_idx")
            variant = scan.get("variant_id", "")
            if stage != last_stage and last_stage is not None:
                emit({"type": "stage", "scan": i, "variant": variant})
            last_stage = stage

            emit({"type": "scan", "n": i, "total": len(scans),
                  "scan_idx": scan["scan_idx"], "variant": variant,
                  "peaks": [{"f": p["freq_mhz"], "w": p["width_mhz"],
                             "p": p["power_dbm"]} for p in scan["detected_peaks"]],
                  "active": [{"cf": g["center_freq"],
                              "bw": g["channel_def"].get("bandwidth_override") or 15.0}
                             for g in scan["ground_truth"] if g["active_this_scan"]]})

            results: dict[str, dict] = {}

            def arm_off():
                user = (f"--- Scan {scan['scan_idx']} ---\n"
                        f"Detected peaks:\n{_fmt_peaks(scan)}\n\n"
                        f"Report all persistent transmitters in the band.")
                text = complete(system=_BASE_SYSTEM, user=user,
                                temperature=0.2, max_output_tokens=1400,
                                json_mode=True)
                iv, count, ok = _parse_report(text)
                results["off"] = {"iv": iv, "count": count, "ok": ok, "known": count}

            def arm_replay():
                # The off arm's prompt plus every earlier scan verbatim.
                # No external memory: the model must rebuild the picture
                # from the raw history on every call.
                if replay_history:
                    hist = "\n\n".join(replay_history)
                    user = (f"EARLIER SCANS (full history so far):\n{hist}\n\n"
                            f"--- Scan {scan['scan_idx']} (current) ---\n"
                            f"Detected peaks:\n{_fmt_peaks(scan)}\n\n"
                            f"Report all persistent transmitters in the band, "
                            f"including ones only seen in earlier scans.")
                else:
                    user = (f"--- Scan {scan['scan_idx']} ---\n"
                            f"Detected peaks:\n{_fmt_peaks(scan)}\n\n"
                            f"Report all persistent transmitters in the band.")
                text = complete(system=_BASE_SYSTEM, user=user,
                                temperature=0.2, max_output_tokens=1400,
                                json_mode=True)
                iv, count, ok = _parse_report(text)
                results["icl"] = {"iv": iv, "count": count, "ok": ok,
                                  "known": count,
                                  "ctx": len(replay_history) + 1}

            def arm_on():
                # 1. Recall the registry from Mubit.
                reg = []
                try:
                    out = mubit.recall(
                        query="bsm-registry transmitter registry",
                        limit=3, entry_types=["lesson"], evidence_only=True,
                        include_working_memory=False, prefer_current_run=True)
                    for e in out.get("evidence") or []:
                        txt = (e.get("content") or e.get("text") or "").strip()
                        if txt.startswith(_REG_TAG):
                            reg = json.loads(txt[len(_REG_TAG):])
                            break
                except Exception:
                    pass
                # 2. Merge the current peaks into it.
                added = _merge(reg, scan, scan["scan_idx"])
                if added:
                    emit({"type": "registry", "arm": "on", "size": len(reg),
                          "added": added, "scan": i})
                # 3. One LLM call with the registry as evidence.
                user = (f"--- Scan {scan['scan_idx']} ---\n"
                        f"Detected peaks:\n{_fmt_peaks(scan)}\n\n"
                        f"ACCUMULATED TRANSMITTER REGISTRY "
                        f"(every transmitter observed across all scans so far):\n"
                        f"{_fmt_registry(reg)}\n\n"
                        f"Report all persistent transmitters in the band, "
                        f"including dormant ones from the registry.")
                text = complete(system=_BASE_SYSTEM, user=user,
                                temperature=0.2, max_output_tokens=1400,
                                json_mode=True)
                iv, count, ok = _parse_report(text)
                results["on"] = {"iv": iv, "count": count, "ok": ok,
                                 "known": len(reg)}
                # 4. Write the updated registry back to Mubit.
                try:
                    mubit.remember(
                        content=f"{_REG_TAG}{json.dumps(reg, separators=(',', ':'))}",
                        intent="lesson", lesson_type="success",
                        lesson_scope="run", lesson_importance="high",
                        upsert_key="bsm-registry",
                        metadata={"task": "bsm"}, wait=True)
                except Exception:
                    pass
                registry[:] = reg

            threads = [
                threading.Thread(target=arm_on, name="arm-on", daemon=True),
                threading.Thread(target=arm_off, name="arm-off", daemon=True),
                threading.Thread(target=arm_replay, name="arm-icl", daemon=True),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # The current scan enters the replay arm's history for the
            # next call (after the joins, so this scan's call did not
            # already contain it).
            replay_history.append(
                f"--- Scan {scan['scan_idx']} ---\n"
                f"Detected peaks:\n{_fmt_peaks(scan)}")

            for arm in ("off", "icl", "on"):
                r = results.get(arm) or {"iv": [], "count": 0, "ok": False,
                                         "known": 0}
                iou = iou_available(r["iv"], truth)
                scores[arm].append(iou)
                emit({"type": "report", "arm": arm, "n": i,
                      "iou": round(iou, 3),
                      "mean_iou": round(sum(scores[arm]) / len(scores[arm]), 3),
                      "count": r["count"], "known": r["known"],
                      "ctx": r.get("ctx"),
                      "parse_ok": r["ok"],
                      "intervals": [[round(a, 2), round(b, 2)] for a, b in r["iv"]]})
            time.sleep(0.4)

        emit({"type": "done", "summary": {
            arm: {"mean_iou": round(sum(s) / len(s), 3) if s else 0,
                  "final_iou": round(s[-1], 3) if s else 0,
                  "scans": len(s)}
            for arm, s in scores.items()}})
