"""In-memory V2 ticket memory store.

This store is process-local and shadow-mode only. It intentionally does not
write to disk, which keeps V1 reproducibility and secret handling unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.ticket_memory import TicketMemory


@dataclass
class MemoryStore:
    _items: dict[int, TicketMemory] = field(default_factory=dict)

    def upsert(self, ticket_id: int, memory: TicketMemory) -> None:
        if not memory.validate():
            raise ValueError("invalid_ticket_memory")
        self._items[int(ticket_id)] = memory

    def get(self, ticket_id: int) -> TicketMemory | None:
        return self._items.get(int(ticket_id))

    def as_dict(self) -> dict[int, dict]:
        return {ticket_id: memory.to_dict() for ticket_id, memory in sorted(self._items.items())}

