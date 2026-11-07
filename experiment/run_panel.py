"""Run the OpenRouter part of the standard panel (panel.json) on every condition:
the six original cases (explicit labels, 5 seeds; name cues, 2 names x 5 seeds) and
the 50 Q-Pain cases (both cues, 3 seeds): 3,120 requests per model configuration.

Records are written in the same formats and keys as the local runs (run.py and
qpain.py), so analyze.py, make_paper_assets.py and audit_analyses.py read them
unchanged. Requests run concurrently; responses already on disk are skipped.

    python3.10 run_panel.py --list
    python3.10 run_panel.py --smoke                 # one request per configuration
    python3.10 run_panel.py --only openai/gpt-4.1   # one configuration
    python3.10 run_panel.py                         # everything, within the budget
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openrouter
import qpain
import run as harness
from analyze import parse as parse_main
from records import write_record

ERRORS = Path(__file__).parent / 'outputs' / 'panel_errors.jsonl'

HERE = Path(__file__).parent
PANEL = json.loads((HERE / 'panel.json').read_text())
MAIN_SCHEMA = {'type': 'object',
               'properties': {'pain_severity': {'type': 'integer', 'minimum': 0, 'maximum': 10},
                              'analgesic': {'type': 'string', 'enum': harness.ANALGESICS},
                              'rationale': {'type': 'string'}},
               'required': ['pain_severity', 'analgesic', 'rationale'], 'additionalProperties': False}
QPAIN_SCHEMA = dict(qpain.SCHEMA, additionalProperties=False)


def think_label(spec: dict) -> str:
    """The thinking label recorded and used in keys, matching the local runs' vocabulary."""
    return {None: 'off', 'off': 'off', 'on': 'on'}.get(spec.get('reasoning'), spec.get('reasoning'))


def model_name(spec: dict) -> str:
    return spec['id']


def folder_for(root: Path, spec: dict) -> Path:
    return root / spec['id'].replace('/', '_').replace(':', '_')


def jobs(spec: dict) -> list[dict]:
    """Every request for one configuration, with the path its record will be saved to."""
    out, think = [], think_label(spec)
    name = model_name(spec)
    vignettes = json.loads((HERE / 'vignettes_v2.json').read_text())
    for names in (False, True):
        for vignette in vignettes:
            for arm in harness.ARMS:
                reps = [(r, n) for r in range(5) for n in range(2)] if names else list(range(5))
                for rep in reps:
                    person = harness.NAMES[arm][rep[1]] if names else None
                    tag = (f'|{person}' if names else '') + (f'|think={think}' if think != 'off' else '')
                    rep_key = f'{rep[0]}-{rep[1]}' if names else rep
                    key = harness.item_key(name, vignette['id'] + tag, arm, rep_key)
                    template = harness.NAMED_PROMPT if names else harness.PROMPT
                    prompt = template.format(analgesics=harness.ANALGESICS, age=vignette['age'], name=person,
                                             race=arm[0], sex=arm[1], case=vignette['case'])
                    suffix = ('_named' if names else '') + (f'_think-{think}' if think != 'off' else '')
                    out.append({'path': folder_for(harness.RAW, spec) / f'{key}.json', 'prompt': prompt,
                                'seed': rep[0] if names else rep, 'schema': MAIN_SCHEMA,
                                'record': {'model': name, 'backend': 'openrouter', 'vignette': vignette['id'] + suffix,
                                           'lang': 'en', 'cue': 'names' if names else 'explicit', 'think': think,
                                           'vignette_set': 'vignettes_v2.json', 'race': arm[0], 'sex': arm[1],
                                           'name': person, 'rep': rep_key}})
    for cue in ('explicit', 'names'):
        for item in qpain.load_vignettes():
            for arm in harness.ARMS:
                for seed in range(3):
                    prompt, person = qpain.build_prompt(item, arm, cue)
                    out.append({'path': folder_for(qpain.RAW, spec) / f"{qpain.key(name, item['id'], arm, cue, seed, think)}.json",
                                'prompt': prompt, 'seed': seed, 'schema': QPAIN_SCHEMA,
                                'record': {'model': name, 'backend': 'openrouter', 'item': item['id'],
                                           'context': item['context'], 'index': item['index'], 'cue': cue,
                                           'think': think, 'race': arm[0], 'sex': arm[1], 'name': person, 'seed': seed}})
    return out


def execute(spec: dict, job: dict, cap: float) -> str:
    if job['path'].exists():
        return 'cached'
    started = time.time()
    result = openrouter.call(spec, job['prompt'], seed=job['seed'], schema=job['schema'], cap_usd=cap)
    record = dict(job['record'], prompt=job['prompt'], response=result['response'], thinking=result['thinking'],
                  provider=result['provider'], cost=result['cost'], usage=result['usage'],
                  seed_sent=spec.get('seed', True), seconds=round(time.time() - started, 2),
                  saved_at=time.strftime('%Y-%m-%dT%H:%M:%S'), panel_spec=spec,
                  request=result['request'], raw_response=result['raw_response'],
                  http=result['http'], attempts=result['attempts'], capture='full-v1')
    record['valid'] = (qpain.parse if 'item' in record else parse_main)(record['response']) is not None
    write_record(job['path'], record)
    return 'called'


def log_error(spec: dict, job: dict, exc: Exception) -> None:
    ERRORS.parent.mkdir(parents=True, exist_ok=True)
    detail = getattr(exc, 'detail', None)
    if detail is not None:                      # keep the failed exchange in full
        failed = ERRORS.parent / 'panel_errors' / job['path'].parent.name / (
            job['path'].stem + time.strftime('_%Y%m%dT%H%M%S') + '.json')
        write_record(failed, dict(job['record'], prompt=job['prompt'], panel_spec=spec,
                                  error=f'{type(exc).__name__}: {exc}', **detail))
    with ERRORS.open('a') as handle:
        handle.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': spec['id'],
                                 'think': think_label(spec), 'path': str(job['path']),
                                 'error': f'{type(exc).__name__}: {str(exc)[:300]}'}) + '\n')


def run_spec(spec: dict, cap: float, workers: int, limit: int | None, log) -> dict:
    todo = [j for j in jobs(spec) if not j['path'].exists()]
    if limit is not None:
        todo = todo[:limit]
    counts = {'called': 0, 'errors': 0, 'skipped_existing': len(jobs(spec)) - len([j for j in jobs(spec) if not j['path'].exists()])}
    label = f"{spec['id']} [{think_label(spec)}]"
    log(f'{label}: {len(todo)} to run, {counts["skipped_existing"]} already on disk')
    stop = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, spec, job, cap): job for job in todo}
        for future in as_completed(futures):
            try:
                if future.result() == 'called':
                    counts['called'] += 1
            except openrouter.BudgetExceeded as exc:
                if not stop:
                    log(f'{label}: {exc}; stopping'); stop = True
                    for f in futures:
                        f.cancel()
            except Exception as exc:
                counts['errors'] += 1
                log_error(spec, futures[future], exc)
                if counts['errors'] <= 5:
                    log(f'{label}: error {str(exc)[:200]}')
                if counts['errors'] >= 25 and counts['called'] == 0:
                    log(f'{label}: 25 errors and no successes; abandoning this configuration')
                    for f in futures:
                        f.cancel()
                    break
            done = counts['called']
            if done and done % 500 == 0:
                log(f'{label}: {done} called, ${openrouter.spent():.2f} spent in total')
    log(f'{label}: {counts}, ${openrouter.spent():.2f} spent in total')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='one request per configuration')
    parser.add_argument('--only', default=None, help='run only configurations whose id contains this text')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--cap', type=float, default=PANEL['budget_usd'])
    args = parser.parse_args()
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", file=sys.stderr, flush=True)
    specs = [s for s in PANEL['openrouter'] if not args.only or args.only in s['id']]
    if args.list:
        for s in specs:
            print(f"{s['id']:38} [{think_label(s):4}] {s['provider']:16} {s.get('quantization', '-'):5} seed={s.get('seed', True)}  {len(jobs(s))} requests")
        sys.exit()
    for spec in specs:
        run_spec(spec, args.cap, 1 if args.smoke else args.workers, 1 if args.smoke else None, log)
    log(f'done; ${openrouter.spent():.2f} spent of the ${args.cap:.2f} cap')
