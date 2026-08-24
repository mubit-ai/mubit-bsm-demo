"""Mubit SDK call tracer — captures every recall()/remember()/handoff() etc.

Monkey-patches ``_Transport.invoke`` (the single chokepoint in the SDK) so that
every Mubit wire call is observed and emitted to the UI transparency pane.

Thread-safe: the tracer callback uses ``call_soon_threadsafe`` so it works from
worker threads (``asyncio.to_thread``) as well as the event loop.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Install the patch ONCE at import time.
# ---------------------------------------------------------------------------

try:
    from mubit.client import _Transport
except ImportError:  # pragma: no cover — SDK not installed
    _Transport = None  # type: ignore[assignment]

_OP_LABELS: dict[str, str] = {
    "control.query": "recall",
    "control.ingest": "remember",
    "control.create_handoff": "handoff",
    "control.submit_feedback": "feedback",
    "control.record_outcome": "outcome",
    "control.record_step_outcome": "step_outcome",
    "control.reflect": "reflect",
    "control.get_context": "context",
    "control.create_checkpoint": "checkpoint",
    "control.register_agent": "register",
    "control.agents_register": "register",
    "control.list_agents": "list_agents",
    "control.receive_handoffs": "receive",
    "control.surface_strategies": "strategy",
    "control.circuit_break": "circuit_break",
}

# Opaque ops we don't care about (health checks, job polling, etc.)
_SKIP_OPS = frozenset({
    "core.health", "control.get_ingest_job", "control.list_sessions",
    "control.create_session",
})

_lock = threading.Lock()
_current_emitter: Optional[Callable[[dict], None]] = None

_installed = False


def _excerpt(obj: Any, maxlen: int = 240) -> Optional[str]:
    """Extract a human-readable excerpt from a payload or result dict."""
    if obj is None:
        return None
    try:
        if isinstance(obj, dict):
            # Query / recall payload
            if "query" in obj:
                return f"query: {str(obj['query'])[:120]}"
            # Ingest payload — extract item text
            items = obj.get("items") or []
            if items and isinstance(items, list):
                first = items[0]
                if isinstance(first, dict):
                    txt = first.get("text") or first.get("content") or ""
                    intent = first.get("intent", "")
                    lt = first.get("lesson_type", "")
                    parts = []
                    if intent:
                        parts.append(intent)
                    if lt:
                        parts.append(lt)
                    prefix = f"({', '.join(parts)}) " if parts else ""
                    return f"{prefix}{str(txt)[:160]}"
            # Handoff payload
            if "content" in obj and "to_agent_id" in obj:
                return f"→ {obj.get('to_agent_id', '?')}: {str(obj['content'])[:120]}"
            # Generic
            return json.dumps(obj, default=str)[:maxlen]
        return str(obj)[:maxlen]
    except Exception:
        return None


def _result_excerpt(result: Any, maxlen: int = 200) -> Optional[str]:
    """Extract a summary from a response."""
    if result is None:
        return None
    try:
        if isinstance(result, dict):
            # Recall response
            ev = result.get("evidence") or []
            if ev:
                count = len(ev)
                first_text = ""
                if ev:
                    first_text = ev[0].get("content") or ev[0].get("text") or ""
                return f"{count} result{'s' if count != 1 else ''}: {str(first_text)[:120]}"
            # Ingest response
            if result.get("done") is not None or result.get("status"):
                return f"status={result.get('status') or result.get('done')}"
            # Handoff response
            if result.get("handoff_id"):
                return f"handoff_id={result['handoff_id'][:16]}"
            return json.dumps(result, default=str)[:maxlen]
        return str(result)[:maxlen]
    except Exception:
        return None


def _semantic_summary(op_key: str, payload: Any, result: Any) -> Optional[str]:
    """Generate a human-readable one-liner for the Mubit I/O pane."""
    try:
        label = _OP_LABELS.get(op_key, op_key)
        if op_key == "control.query" and isinstance(payload, dict) and isinstance(result, dict):
            n = len(result.get("evidence") or [])
            q = str(payload.get("query", ""))[:60]
            return f"recall: {n} lesson{'s' if n != 1 else ''} for \"{q}\""
        if op_key == "control.ingest" and isinstance(payload, dict):
            items = payload.get("items") or []
            if items:
                it = items[0]
                txt = str(it.get("text") or it.get("content") or "")[:80]
                return f"remember: {txt}"
        if op_key == "control.create_handoff" and isinstance(payload, dict):
            to = payload.get("to_agent_id", "?")
            frm = payload.get("from_agent_id", "?")
            content = str(payload.get("content", ""))[:60]
            return f"handoff: {frm} → {to}: {content}"
        if op_key == "control.submit_feedback" and isinstance(payload, dict):
            v = payload.get("verdict", "?")
            return f"feedback: {v.upper()}"
        if op_key == "control.record_outcome" and isinstance(payload, dict):
            outcome = payload.get("outcome", "?")
            conf = ""
            if isinstance(result, dict):
                uc = result.get("updated_confidence")
                rc = result.get("reinforcement_count")
                if uc is not None:
                    conf = f" → confidence={uc:.2f}, reinforcement=#{rc}"
            return f"outcome: {outcome}{conf}"
        if "step_outcome" in op_key and isinstance(payload, dict):
            sig = payload.get("signal", "?")
            step = payload.get("step_name") or payload.get("step_id", "?")
            return f"step: {step} (signal={sig})"
        if "register" in op_key and isinstance(payload, dict):
            aid = payload.get("agent_id", "?")
            return f"register: {aid}"
        if op_key == "control.reflect":
            stored = 0
            degraded = False
            if isinstance(result, dict):
                stored = result.get("lessons_stored", 0)
                degraded = result.get("degraded", False)
            tag = " (degraded)" if degraded else ""
            return f"reflect: {stored} lessons{tag}"
        if "strategy" in op_key:
            n = 0
            if isinstance(result, dict):
                n = len(result.get("strategies", []))
            return f"strategy: {n} emerged"
        if "checkpoint" in op_key and isinstance(payload, dict):
            lbl = payload.get("label", "unnamed")
            return f"checkpoint: {lbl}"
    except Exception:
        pass
    return None


def _make_traced_invoke(real_invoke):
    """Wrap ``_Transport.invoke`` with call observation."""

    def traced(self, op, payload=None, *, transport=None):
        op_key = op.get("key", "unknown")
        if op_key in _SKIP_OPS:
            return real_invoke(self, op, payload, transport=transport)

        started = time.perf_counter()
        err = None
        result = None
        try:
            result = real_invoke(self, op, payload, transport=transport)
            return result
        except Exception as e:
            err = e
            raise
        finally:
            with _lock:
                emitter = _current_emitter
            if emitter is not None:
                latency = round((time.perf_counter() - started) * 1000, 1)
                http = op.get("http", {})
                emitter({
                    "type": "mubit",
                    "op": op_key,
                    "label": _OP_LABELS.get(op_key, op_key),
                    "method": http.get("method", ""),
                    "path": http.get("path", ""),
                    "latency_ms": latency,
                    "semantic": _semantic_summary(op_key, payload, result),
                    "payload_excerpt": _excerpt(payload),
                    "result_excerpt": _result_excerpt(result),
                    "error": str(err) if err else None,
                })

    return traced


def install():
    """Install the monkey-patch (idempotent)."""
    global _installed
    if _installed or _Transport is None:
        return
    _Transport.invoke = _make_traced_invoke(_Transport.invoke)
    _installed = True


def set_emitter(emitter: Optional[Callable[[dict], None]]):
    """Set the active emitter callback (called from any thread)."""
    global _current_emitter
    with _lock:
        _current_emitter = emitter


def clear_emitter():
    """Clear the active emitter."""
    global _current_emitter
    with _lock:
        _current_emitter = None


# Auto-install on import.
install()
