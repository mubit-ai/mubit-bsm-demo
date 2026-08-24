# Mubit BSM Sample Demo

A three-arm demo over a small sample of the Continual Learning Bench
blind-spectrum-monitoring task. Each scan of a 168 MHz radio band shows
only the transmitters that are active at that moment — about a quarter
of the truth. The task is to report all persistent transmitters, dormant
ones included.

Three arms process the same scans with the same model, in lockstep. They
differ only in what they carry between scans:

- **Memory off** holds nothing. Its score stays flat, and it makes the
  cheapest calls.
- **Context replay** holds no external memory but resends every earlier
  scan verbatim in each prompt. Its score climbs, and its per-scan cost
  grows with the stream because the prompt contains the full history.
- **Memory on** keeps a transmitter registry in Mubit: `recall` before
  each report, `remember` (upsert) after it. Its score climbs as the
  registry accumulates, and the registry block in the prompt has a
  bounded size (one line per known transmitter) regardless of stream
  length.

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

Press **Run Sample**. The run makes three LLM calls per scan (one per
arm) plus the on arm's Mubit calls. `BSM_LO` / `BSM_HI` change the scan
window; `DEMO_PORT` changes the port.

## What the UI shows

- **The band strip** — the window's transmitters (bright = active this
  scan, dim = dormant), and each arm's latest report below it. The on
  arm's row keeps blocks that vanished from the current scan; only the
  registry carries them.
- **Score per scan** — each arm's availability IoU, live. The off arm is
  flat; the replay and on arms climb.
- **Cumulative LLM spend** — each arm's running cost. Off and on grow
  linearly; the replay curve bends upward as its prompt grows.
- **Learning ledger** — every registry addition, and the drift boundary.
- **End card** — mean and final IoU, transmitters known, calls, tokens,
  and spend per arm, with the spend-shape and measurement notes.

This page is a demo protocol on a small sample. The leaderboard numbers
come from the benchmark's own harness on the full 90-scan schedule.
