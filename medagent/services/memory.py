from dataclasses import dataclass, field


@dataclass
class MemoryStore:
    short_term: dict[str, list[str]] = field(default_factory=dict)
    long_term: dict[str, dict[str, str]] = field(default_factory=dict)
    episodic: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def append_turn(self, user_id: str, text: str) -> None:
        turns = self.short_term.setdefault(user_id, [])
        turns.append(text)
        if len(turns) > 10:
            self.short_term[user_id] = turns[-10:]

    def get_recent(self, user_id: str) -> list[str]:
        return self.short_term.get(user_id, [])

    def upsert_profile_fact(self, user_id: str, key: str, value: str) -> None:
        profile = self.long_term.setdefault(user_id, {})
        profile[key] = value

    def get_profile(self, user_id: str) -> dict[str, str]:
        return self.long_term.get(user_id, {})

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
                1 if query[:2] and query[:2] in x["content"].lower() else 0
            ) + (1 if query[:2] and query[:2] in x["topic"].lower() else 0),
            reverse=True,
        )
        return ranked[:top_k]
