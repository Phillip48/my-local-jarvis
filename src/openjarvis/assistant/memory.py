"""Long-term memory: a list of facts about the user, persisted as JSON."""

import json
import threading
from datetime import date

from . import config

_lock = threading.Lock()


def load() -> list[dict]:
    if not config.MEMORY_FILE.exists():
        return []
    try:
        return json.loads(config.MEMORY_FILE.read_text())
    except json.JSONDecodeError:
        return []


def _save(facts: list[dict]) -> None:
    config.MEMORY_FILE.write_text(json.dumps(facts, indent=2))


def remember(fact: str) -> str:
    with _lock:
        facts = load()
        if any(f["fact"].lower() == fact.lower() for f in facts):
            return "Already known."
        facts.append({"fact": fact, "added": date.today().isoformat()})
        _save(facts)
    return "Saved to memory."


def forget(query: str) -> str:
    with _lock:
        facts = load()
        kept = [f for f in facts if query.lower() not in f["fact"].lower()]
        removed = len(facts) - len(kept)
        _save(kept)
    return f"Removed {removed} memory item(s)." if removed else "Nothing matched."


def as_prompt() -> str:
    facts = load()
    if not facts:
        return "You don't know anything about the user yet."
    return "\n".join(f"- {f['fact']}" for f in facts)
