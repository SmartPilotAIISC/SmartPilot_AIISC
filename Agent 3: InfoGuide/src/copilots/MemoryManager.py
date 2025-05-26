import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any

MEMORY_PATH = "memory.json"


def load_memory() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_PATH):
        return {
            "user_profile": {},
            "semantic_memory": {},
            "episodic_memory": {
                "causal_discovery": [],
                "rca_runs": []
            },
            "procedural_memory": {}
        }
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory: Dict[str, Any]) -> None:
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)


def save_event_to_memory(event_type: str, metadata: Dict[str, Any]) -> None:
    memory = load_memory()
    metadata["timestamp"] = datetime.now().isoformat()

    if event_type == "causal_discovery":
        memory["episodic_memory"].setdefault("causal_discovery", []).append(metadata)

    elif event_type == "rca_run":
        memory["episodic_memory"].setdefault("rca_runs", []).append(metadata)

    elif event_type == "semantic_annotation":
        memory["semantic_memory"].setdefault("sensor_annotations", {}).update(metadata)

    elif event_type == "procedural_preference":
        memory["procedural_memory"].update(metadata)

    elif event_type == "user_profile":
        memory["user_profile"].update(metadata)

    else:
        raise ValueError(f"Unsupported event type: {event_type}")

    save_memory(memory)


def get_recent_rca_runs(n: int = 5) -> List[Dict[str, Any]]:
    memory = load_memory()
    return memory.get("episodic_memory", {}).get("rca_runs", [])[-n:]


def get_recent_causal_graphs(n: int = 5) -> List[Dict[str, Any]]:
    memory = load_memory()
    return memory.get("episodic_memory", {}).get("causal_discovery", [])[-n:]


def get_semantic_annotation(sensor: str) -> Optional[str]:
    memory = load_memory()
    return memory.get("semantic_memory", {}).get("sensor_annotations", {}).get(sensor)


def get_user_preferences() -> Dict[str, Any]:
    memory = load_memory()
    return memory.get("procedural_memory", {})


def get_user_profile() -> Dict[str, Any]:
    memory = load_memory()
    return memory.get("user_profile", {})


def enrich_context_with_memory(context: str, user_input: str) -> str:
    memory = load_memory()
    mem_lines = []

    if memory.get("procedural_memory"):
        mem_lines.append("⚙️ Preferences:\n" + "\n".join([f"- {k}: {v}" for k, v in memory["procedural_memory"].items()]))

    if memory.get("semantic_memory", {}).get("sensor_annotations"):
        annots = memory["semantic_memory"]["sensor_annotations"]
        mem_lines.append("📘 Sensor Annotations:\n" + "\n".join([f"- {k}: {v}" for k, v in annots.items()]))

    if memory.get("episodic_memory", {}).get("causal_discovery"):
        latest = memory["episodic_memory"]["causal_discovery"][-3:]
        mem_lines.append("🕓 Recent Causal Graphs:\n" + "\n".join([f"- Features: {d['features']}" for d in latest]))

    if mem_lines:
        context += "\n\n---\n📚 Memory Context:\n" + "\n\n".join(mem_lines)

    return context
