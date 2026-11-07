"""Atomic record writes: a crash mid-write never leaves a truncated record on disk."""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)          # atomic on the same filesystem
