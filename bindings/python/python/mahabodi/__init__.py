"""MahaBodi: a System-1 engine for AI agents.

    from mahabodi import Bodi
    b = Bodi()
    b.ingest(open("kb.md").read(), source="kb")
    r = b.query("refund policy")          # r["matched"], r["handoff"], r["stage"], r["hits"]
    b.load_laya("models/laya-v2")         # optional: Laya System-1 decisions via ONNX Runtime
    b.decide("I was charged twice", {"refund": {"type": "noul", "instructions": "Asks for a refund?"}})

Every method returns plain dicts/lists (JSON). Errors raise ValueError.
"""
import json
from typing import Any, Dict, List, Optional, Union

from ._mahabodi import Engine, __version__

__all__ = ["Bodi", "__version__"]

State = Union[str, dict, list]


class Bodi:
    def __init__(self, config: Optional[Union[dict, str]] = None):
        if isinstance(config, dict):
            config = json.dumps(config)
        self._e = Engine(config or "")

    def call(self, method: str, **args: Any) -> Any:
        return json.loads(self._e.call_json(method, json.dumps(args, ensure_ascii=False)))

    # --- memory -------------------------------------------------------------------
    def ingest(self, text: str, format: str = "auto", source: str = "doc") -> dict:
        return self.call("ingest", text=text, format=format, source=source)

    def ingest_batch(self, docs: List[dict]) -> dict:
        """Bulk ingest [{"text", "format"?, "source"?}, ...] with a single rebuild (use for many documents)."""
        return self.call("ingest_batch", docs=docs)

    def query(self, q: str, k: int = 5) -> dict:
        return self.call("query", q=q, k=k)

    def traverse(self, start: str, hops: int = 2, limit: int = 50) -> dict:
        return self.call("traverse", start=start, hops=hops, limit=limit)

    def context(self, q: str, k: int = 5, max_chars: int = 2000) -> dict:
        return self.call("context", q=q, k=k, max_chars=max_chars)

    def density(self) -> dict:
        return self.call("density")

    def ensure_density(self) -> dict:
        return self.call("ensure_density")

    def fastmemory_search(self, q: str) -> list:
        return self.call("fastmemory_search", q=q)

    def snapshot(self) -> dict:
        return self.call("snapshot")

    def restore(self, snapshot: dict) -> dict:
        return self.call("restore", snapshot=snapshot)

    def clear(self) -> None:
        self.call("clear")

    def stats(self) -> dict:
        return self.call("stats")

    # --- System 1 (Laya) ------------------------------------------------------------
    def load_laya(self, dir: str, **options: Any) -> None:
        self.call("load_laya", dir=dir, options=options)

    def predict(self, state: State, questions: Dict[str, dict]) -> dict:
        """Laya-compatible prediction (same maths and output schema as laya.Agent.predict)."""
        return self.call("predict", state=state, questions=questions)

    def learn(self, states: List[State], questions: Dict[str, dict], labels: List[Dict[str, Any]], calibrate: int = 0) -> dict:
        """Store labelled cases as experience. labels[i]: {question_id: gold}.
        calibrate=N: also run Laya on up to N of these cases and compare with the memory's
        leave-one-out accuracy; where memory is clearly better, decide() answers from memory."""
        return self.call("learn", states=states, questions=questions, labels=labels, calibrate=calibrate)

    def decide_defaults(self) -> dict:
        """The engine's effective DecideOptions defaults (for provenance in benchmarks)."""
        return self.call("decide_defaults")

    def load_embedder(self, dir: str, **options: Any) -> None:
        """Dense text embedder (research/export_embedder.py): experience memory uses it when loaded."""
        self.call("load_embedder", dir=dir, options=options)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        return self.call("embed_text", texts=texts)

    def embed(self, states: List[State], questions: Dict[str, dict]) -> List[List[List[float]]]:
        """Pooled decision-context vectors, states x questions (needs a `pooled` model export)."""
        return self.call("embed", states=states, questions=questions)

    def forget(self) -> None:
        self.call("forget")

    def decide(self, state: State, questions: Dict[str, dict], **options: Any) -> dict:
        return self.call("decide", state=state, questions=questions, options=options or None)

    def decide_batch(self, states: List[State], questions: Dict[str, dict], **options: Any) -> List[dict]:
        return self.call("decide_batch", states=states, questions=questions, options=options or None)

    def decide_with_memory(self, state: State, questions: Dict[str, dict], query: Optional[str] = None,
                           k: int = 3, max_chars: int = 1200, style: Optional[Union[str, dict]] = None, **options: Any) -> dict:
        """style: "passages" (top-3 passage texts under `passage`, first; best on BoolQ dev), "labelled"
        (default), or {"passages_only": bool, "top": int, "key": str, "first": bool}."""
        args = dict(state=state, questions=questions, k=k, max_chars=max_chars, options=options or None, style=style)
        if query is not None:
            args["query"] = query
        return self.call("decide_with_memory", **args)
