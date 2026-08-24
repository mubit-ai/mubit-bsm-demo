"""BSM sample demo server — port 7873.

Serves the three-arm view (memory off / context replay / memory on) over
real Continual Learning Bench scans.
Run ./run_demo.sh from the repository root, then open
http://localhost:7873
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from demo.mubit_tracer import set_emitter, clear_emitter  # noqa: E402
from demo.llm_stream import set_usage_emitter, clear_usage_emitter  # noqa: E402
from demo.scenario import BSMScenario  # noqa: E402

app = FastAPI(title="Mubit BSM Sample Demo")
_STATIC_DIR = _HERE / "static"


def _arm_of_thread() -> str | None:
    name = threading.current_thread().name
    if name == "arm-on":
        return "on"
    if name == "arm-off":
        return "off"
    if name == "arm-icl":
        return "icl"
    return None


@app.get("/", response_class=HTMLResponse)
async def page():
    html_path = _STATIC_DIR / "bsm.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>bsm.html not found</h1>", status_code=404)


class RunRequest(BaseModel):
    scenario: str = "bsm"


@app.post("/api/run")
async def run(req: RunRequest):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def emit(event: dict) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            pass

    async def stream():
        model = os.environ.get("DEMO_MODEL", "gemini-3.7-flash")
        yield f"data: {json.dumps({'type': 'start', 'model': model})}\n\n"

        def _tag(e: dict) -> dict:
            arm = _arm_of_thread()
            return {**e, "arm": arm} if arm else e

        set_emitter(lambda e: emit(_tag(e)))
        set_usage_emitter(lambda u: emit(_tag({"type": "llm_call", **u})))

        def _run():
            try:
                BSMScenario().run(emit)
            except Exception as e:
                emit({"type": "error", "message": f"Scenario error: {e}"})
            finally:
                emit({"type": "run_finished"})

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                if not thread.is_alive():
                    break
                continue
            if event.get("type") == "run_finished":
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"

        thread.join(timeout=10)
        clear_emitter()
        clear_usage_emitter()
        yield f"data: {json.dumps({'type': 'all_done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7873)
