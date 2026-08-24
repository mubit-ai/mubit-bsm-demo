# Mubit BSM Sample Demo

A memory on-vs-off demo over a small sample of the Continual Learning
Bench blind-spectrum-monitoring task. Each scan of a 168 MHz radio band
shows only the transmitters that are active at that moment — about a
quarter of the truth. The task is to report all persistent transmitters,
dormant ones included.

Two arms process the same scans with the same model, in lockstep:

- **Memory off** holds nothing between scans. Its score stays flat.
- **Memory on** keeps a transmitter registry in Mubit: `recall` before
  each report, `remember` (upsert) after it. Its score climbs as the
  registry accumulates.

Scoring is the benchmark's published availability-IoU rule, reimplemented
here over the stages present in the sampled window. The default window
(scans 19-42 of the `mixed_grid_lifecycle` corpus) crosses one
concept-drift boundary: new narrowband channels start appearing mid-run.

The scan data is read in place from a local clone of the benchmark
repository and is not copied here.

## What you need

- Python 3.11 or later.
- A running Mubit instance and an API key.
- A Gemini API key, or an OpenAI API key for `gpt-*` models.
- A clone of the benchmark repository at
  `~/code/continual-learning-bench` (or set `BSM_DATA` to the
  `mixed_grid_lifecycle.jsonl` path).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env     # fill in the values

./run_demo.sh            # then open http://localhost:7873
```

Press **Run Sample**. The run makes two LLM calls per scan (one per arm)
plus the on arm's Mubit calls. `BSM_LO` / `BSM_HI` change the scan
window; `DEMO_PORT` changes the port.

## What the UI shows

- **The band strip** — the window's transmitters (bright = active this
  scan, dim = dormant), and each arm's latest report below it. The on
  arm's row keeps blocks that vanished from the current scan; only the
  registry carries them.
- **Score per scan** — both arms' availability IoU, live. The off arm is
  flat; the on arm climbs.
- **Learning ledger** — every registry addition, and the drift boundary.
- **End card** — mean and final IoU, transmitters known, calls, tokens,
  and spend per arm, with the measurement notes.

This page is a demo protocol on a small sample. The leaderboard numbers
come from the benchmark's own harness on the full 90-scan schedule.
