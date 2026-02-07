from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IpcMessage:
    type: str
    id: int | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    text: str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_json_line(self) -> str:
        payload: dict[str, Any] = {"type": self.type}
        if self.id is not None:
            payload["id"] = self.id
        if self.method is not None:
            payload["method"] = self.method
        if self.params is not None:
            payload["params"] = self.params
        if self.text is not None:
            payload["text"] = self.text
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False) + "\n"


def parse_json_line(line: str) -> dict[str, Any]:
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise TypeError("IPC line must decode to a JSON object")
    return obj

