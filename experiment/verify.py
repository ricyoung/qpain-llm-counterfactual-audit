"""Check every saved response and set aside the bad ones so a rerun replaces them.

For each model folder under outputs/raw and outputs/raw_qpain it counts records,
corrupt files (not valid JSON, e.g. a write cut off by a crash), and responses that
do not parse as a valid answer. With --requeue, corrupt and invalid records are
MOVED (never deleted) to outputs/quarantine/<timestamp>/, preserving their path;
the resumable runners (run.py, qpain.py, run_panel.py) then re-request exactly those,
because a missing file is what marks a request as not done. Leftover *.tmp files
from an interrupted write are quarantined the same way.

The OpenRouter error log (outputs/panel_errors.jsonl) is summarised too; failed
requests were never written, so the next run_panel.py run retries them.

    python3.10 verify.py              # report only
    python3.10 verify.py --requeue    # quarantine bad records for re-request
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import qpain
from analyze import parse as parse_main

HERE = Path(__file__).parent
ROOTS = {'main': HERE / 'outputs' / 'raw', 'qpain': HERE / 'outputs' / 'raw_qpain'}
ERRORS = HERE / 'outputs' / 'panel_errors.jsonl'


def scan() -> tuple[dict, list[Path]]:
    summary, bad = defaultdict(Counter), []
    for kind, root in ROOTS.items():
        for path in sorted(root.glob('*/*')):
            key = f'{kind:5} {path.parent.name}'
            if path.name.endswith('.tmp'):
                summary[key]['partial_write'] += 1; bad.append(path); continue
            if path.suffix != '.json':
                continue
            try:
                record = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                summary[key]['corrupt'] += 1; bad.append(path); continue
            parsed = (qpain.parse if kind == 'qpain' else parse_main)(record.get('response', ''))
            summary[key]['records'] += 1
            if parsed is None:
                summary[key]['invalid'] += 1; bad.append(path)
    return summary, bad


def main(requeue: bool) -> None:
    summary, bad = scan()
    print(f"{'set':5} {'model folder':42} {'records':>8} {'invalid':>8} {'corrupt':>8} {'partial':>8}")
    for key in sorted(summary):
        c = summary[key]
        print(f"{key:48} {c['records']:8} {c['invalid']:8} {c['corrupt']:8} {c['partial_write']:8}")
    if ERRORS.exists():
        errors = [json.loads(line) for line in ERRORS.read_text().splitlines() if line.strip()]
        by_model = Counter(f"{e['model']} [{e['think']}]" for e in errors)
        print(f'\nOpenRouter request errors logged: {len(errors)} (retried automatically on the next run)')
        for model, n in by_model.most_common():
            last = next(e['error'] for e in reversed(errors) if f"{e['model']} [{e['think']}]" == model)
            print(f'  {model:45} {n:5}  last: {last[:90]}')
    print(f'\nrecords needing re-request: {len(bad)}')
    if requeue and bad:
        dest_root = HERE / 'outputs' / 'quarantine' / time.strftime('%Y%m%dT%H%M%S')
        for path in bad:
            dest = dest_root / path.relative_to(HERE / 'outputs')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), dest)
        print(f'moved {len(bad)} to {dest_root} (kept); the next run re-requests them')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--requeue', action='store_true')
    main(parser.parse_args().requeue)
