from dataclasses import dataclass, field
from typing import Any


LIST_LIKE_KEYS = {
    "allergies",
    "chronic_history",
    "current_meds",
    "family_history",
    "surgical_history",
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text]


@dataclass
class MemoryStore:
    short_term: dict[str, list[str]] = field(default_factory=dict)
    long_term: dict[str, dict[str, Any]] = field(default_factory=dict)
    episodic: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    visit_records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    source_documents: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    active_visits: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def append_turn(self, user_id: str, text: str) -> None:
        turns = self.short_term.setdefault(user_id, [])
        turns.append(text)
        if len(turns) > 10:
            self.short_term[user_id] = turns[-10:]

    def get_recent(self, user_id: str) -> list[str]:
        return self.short_term.get(user_id, [])

    def upsert_profile_fact(self, user_id: str, key: str, value: Any) -> None:
        profile = self.long_term.setdefault(user_id, {})
        profile[key] = value

    def append_profile_item(self, user_id: str, key: str, value: str) -> None:
        profile = self.long_term.setdefault(user_id, {})
        items = _as_list(profile.get(key))
        value = value.strip()
        if value and value not in items:
            items.append(value)
        profile[key] = items

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return dict(self.long_term.get(user_id, {}))

    def get_profile_list(self, user_id: str, key: str) -> list[str]:
        return _as_list(self.long_term.get(user_id, {}).get(key))

    def build_clinical_snapshot(self, user_id: str) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        for key in LIST_LIKE_KEYS:
            if key in profile:
                profile[key] = _as_list(profile.get(key))
        return profile

    def append_episode(self, user_id: str, topic: str, content: str) -> None:
        items = self.episodic.setdefault(user_id, [])
        items.append({"topic": topic, "content": content})
        if len(items) > 50:
            self.episodic[user_id] = items[-50:]

    def recall_episodes(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, str]]:
        query = query.lower()
        items = self.episodic.get(user_id, [])
        ranked = sorted(
            items,
            key=lambda x: (
                1 if query[:4] and query[:4] in x["content"].lower() else 0
            ) + (1 if query[:4] and query[:4] in x["topic"].lower() else 0),
            reverse=True,
        )
        return ranked[:top_k]

    def append_visit_record(self, user_id: str, record: dict[str, Any]) -> None:
        items = self.visit_records.setdefault(user_id, [])
        items.append(record)
        if len(items) > 100:
            self.visit_records[user_id] = items[-100:]

    def get_visit_records(self, user_id: str) -> list[dict[str, Any]]:
        return list(self.visit_records.get(user_id, []))

    def append_source_document(self, user_id: str, document: dict[str, Any]) -> None:
        items = self.source_documents.setdefault(user_id, [])
        items.append(document)
        if len(items) > 100:
            self.source_documents[user_id] = items[-100:]

    def get_source_documents(self, user_id: str) -> list[dict[str, Any]]:
        return list(self.source_documents.get(user_id, []))

    def save_active_visit(self, user_id: str, visit_id: str, payload: dict[str, Any]) -> None:
        visits = self.active_visits.setdefault(user_id, {})
        visits[visit_id] = payload

    def get_active_visit(self, user_id: str, visit_id: str) -> dict[str, Any] | None:
        return self.active_visits.get(user_id, {}).get(visit_id)

    def remove_active_visit(self, user_id: str, visit_id: str) -> None:
        visits = self.active_visits.get(user_id, {})
        visits.pop(visit_id, None)
        if not visits and user_id in self.active_visits:
            self.active_visits.pop(user_id, None)
