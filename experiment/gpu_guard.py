#!/usr/bin/env python3.10
"""Keep each local study model fully on the GPU, without splitting a case across placements.

Ollama decides how much of a model goes to the GPU when it loads it, and keeps that placement
for as long as the model stays loaded. An hourly embedding job shared the study
host. If its embedding model holds memory when a study model loads, the study
model lands partly on the CPU and runs several times slower: qwen3:14b loaded at 00:25:35 on
2026-11-06, during the embedding run of 00:22:57-00:38:09, with 5.0 of 14.9 GB on the GPU, at
13 s a call against gemma3's 1.9 s.

The guard waits until no embedding model is loaded, then asks Ollama to unload the study model
while the last call of a case is in flight. That call finishes under the old placement, the model
unloads, and the next case reloads it on the GPU, so every within-case identity comparison runs
under one placement. Each action is logged with the case it fell after. It never touches a model
outside the study list and gives up on a model that is still split after two reloads (then the
memory is held by something else and a reload would not help).

    python3.10 gpu_guard.py            # runs until the local queue finishes
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / 'outputs' / 'gpu_guard.log'
STUDY = {'gemma3:12b', 'qwen3:14b', 'ministral-3:14b', 'llama3.2:latest',
         'qwen3.8:27b', 'gpt-oss:20b', 'qwen3.6:35b'}
ARMS = [(r, s) for r in ('White', 'Black', 'Hispanic', 'Asian') for s in ('man', 'woman')]
QPAIN_SEEDS, REPS, NAME_SLOTS = 3, 5, 2
ON_GPU = 0.97      # share of the model in VRAM that counts as fully placed
MAX_RELOADS = 2


def log(message: str) -> None:
    line = f'{datetime.datetime.now():%F %T} {message}'
    with LOG.open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def call(path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(HOST + path, data=data,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read() or b'{}')


def queue_running() -> bool:
    found = subprocess.run(['pgrep', '-f', 'run_[q]ueue.sh|run_[s]tage.sh|qpain[.]py|run[.]py --vig'],
                           capture_output=True, text=True)
    return bool(found.stdout.strip())


def embed_job_active() -> bool:
    state = subprocess.run(['systemctl', '--user', 'is-active', 'atlas-embed.service'],
                           capture_output=True, text=True)
    return state.stdout.strip() in ('active', 'activating')


def newest_record(model: str) -> dict | None:
    folder = model.replace(':', '_').replace('/', '_')
    files = [f for root in ('raw_qpain', 'raw') for f in (HERE / 'outputs' / root / folder).glob('*.json')]
    if not files:
        return None
    try:
        return json.loads(max(files, key=lambda f: f.stat().st_mtime).read_text())
    except (OSError, ValueError):
        return None


def calls_left_in_case(record: dict) -> int:
    """Calls still to run in the record's case, from the run order case -> arm -> repetition."""
    arm = ARMS.index((record['race'], record['sex']))
    if 'item' in record:                                   # Q-Pain: seeds 0..2 per arm
        return 8 * QPAIN_SEEDS - 1 - (arm * QPAIN_SEEDS + int(record['seed']))
    rep = str(record['rep'])
    if '-' in rep:                                         # authored, name cues: (rep, slot)
        r, n = (int(x) for x in rep.split('-'))
        return 8 * REPS * NAME_SLOTS - 1 - (arm * REPS * NAME_SLOTS + r * NAME_SLOTS + n)
    return 8 * REPS - 1 - (arm * REPS + int(rep))          # authored, explicit labels


def where(record: dict) -> str:
    case = record.get('item') or record.get('vignette')
    return f"{case} {record.get('cue') or ''} {record['race']} {record['sex']} rep {record.get('seed', record.get('rep'))}"


def unload_at_case_end(name: str) -> bool:
    """Wait for the last call of the current case to be in flight, then request the unload."""
    deadline = time.time() + 3600
    while time.time() < deadline:
        record = newest_record(name)
        if record and record.get('model') == name and calls_left_in_case(record) == 1:
            call('/api/generate', {'model': name, 'keep_alive': 0})
            log(f'unload requested for {name} after {where(record)}; the next case reloads it')
            return True
        time.sleep(0.5)
    log(f'no case boundary for {name} within an hour; leaving it')
    return False


def main() -> int:
    reloads: dict[str, int] = {}
    log('guard started')
    while queue_running():
        try:
            models = call('/api/ps').get('models', [])
        except OSError as error:
            log(f'ps failed: {error}')
            time.sleep(30)
            continue
        embedding = any('embed' in m['name'] for m in models)
        for m in models:
            name = m['name']
            if name not in STUDY or not m.get('size'):
                continue
            share = m.get('size_vram', 0) / m['size']
            if share >= ON_GPU:
                continue
            if embedding or embed_job_active():
                continue                                   # memory still held; try after it frees
            if reloads.get(name, 0) >= MAX_RELOADS:
                continue
            reloads[name] = reloads.get(name, 0) + 1
            log(f'{name}: {share:.0%} on GPU ({m["size_vram"]/1e9:.1f} of {m["size"]/1e9:.1f} GB); '
                f'reload {reloads[name]} of {MAX_RELOADS}')
            unload_at_case_end(name)
        time.sleep(15)
    log('local queue finished; guard stopped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
