from __future__ import annotations
from pathlib import Path
import json
from typing import Iterable

def read_jsonl(path: str | Path) -> list[dict]:
    rows=[]
    with Path(path).open() as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
