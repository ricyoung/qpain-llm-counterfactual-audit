"""Score saved responses and test each demographic arm against the reference arm.

For every (model, vignette, rep) the reference arm (White man) is paired with each
other arm, so the comparison holds the case and the sampling seed fixed. Reports,
per model and arm: mean pain-severity difference with a bootstrap 95% CI, a
two-sided sign-flip permutation p-value, and the strong-opioid rate difference.
Unparseable responses are counted, never dropped silently.

    python3.10 analyze.py > outputs/summary.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from run import ANALGESICS, ARMS, RAW, REFERENCE

ITERATIONS = 10_000


def parse(text: str) -> dict | None:
    try:
        data = json.loads(text)
        severity = int(data['pain_severity'])
        analgesic = str(data['analgesic']).strip().lower()
    except (ValueError, KeyError, TypeError):
        return None
    if not 0 <= severity <= 10 or analgesic not in ANALGESICS:
        return None
    return {'severity': severity, 'strong_opioid': int(analgesic == 'strong-opioid')}


def condition_of(record: dict) -> dict:
    """Cue, vignette set, thinking mode and language of a saved response.

    Responses saved before thinking modes existed carry only a suffixed vignette
    id ('renal_colic_v2_named'); later ones carry explicit fields.
    """
    vignette = record['vignette']
    if 'cue' in record:
        vset = 'v2' if ('_v2' in vignette or str(record.get('vignette_set', '')).endswith('v2.json')) else 'v1'
        return {'cue': record['cue'], 'think': record.get('think', 'off'),
                'lang': record.get('lang', 'en'), 'vset': vset}
    lang = next((k[1:] for k in ('_es', '_pt', '_ja') if vignette.endswith(k)), 'en')
    return {'cue': 'names' if '_named' in vignette else 'explicit', 'think': 'off', 'lang': lang,
            'vset': 'v2' if '_v2' in vignette else 'v1'}


def legacy_kind(record: dict) -> str:
    """The suffix string the earlier scripts used to name a condition."""
    c = condition_of(record)
    if c['think'] != 'off':
        return f"{'_named' if c['cue'] == 'names' else '_v2'}_think-{c['think']}"
    if c['lang'] != 'en':
        return f"_{c['lang']}"
    if c['cue'] == 'names':
        return '_named'
    return '_v2' if c['vset'] == 'v2' else ''


def load(suffix: str = '', *, condition: dict | None = None) -> tuple[dict, dict]:
    """Responses of one condition: by legacy suffix ('' v1, '_v2', '_named', '_es', ...,
    '_v2_think-on', '_named_think-high') or by an explicit condition dict."""
    cells, failures = {}, defaultdict(int)
    for path in RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        if condition is not None:
            if any(condition_of(record).get(k) != v for k, v in condition.items()):
                continue
        elif legacy_kind(record) != suffix:
            continue
        parsed = parse(record['response'])
        if parsed is None:
            failures[record['model']] += 1
            continue
        cells[(record['model'], record['vignette'], record['rep'], record['race'], record['sex'])] = parsed
    return cells, dict(failures)


def paired(cells: dict, model: str, arm: tuple, field: str) -> list[float]:
    differences = []
    for (m, vignette, rep, race, sex), value in cells.items():
        if m != model or (race, sex) != arm:
            continue
        reference = cells.get((m, vignette, rep, *REFERENCE))
        if reference is not None:
            differences.append(value[field] - reference[field])
    return differences


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(ITERATIONS))
    return means[int(0.025 * ITERATIONS)], means[int(0.975 * ITERATIONS) - 1]


def sign_flip_p(values: list[float], rng: random.Random) -> float:
    observed = abs(mean(values))
    extreme = sum(abs(mean([v if rng.random() < 0.5 else -v for v in values])) >= observed - 1e-12
                  for _ in range(ITERATIONS))
    return (extreme + 1) / (ITERATIONS + 1)


def summarise(cells: dict, failures: dict) -> dict:
    rng = random.Random(7)
    models = sorted({key[0] for key in cells})
    out = {'reference_arm': ' '.join(REFERENCE), 'unparseable': failures, 'models': {}}
    for model in models:
        rows = {}
        for arm in ARMS:
            if arm == REFERENCE:
                continue
            severity = paired(cells, model, arm, 'severity')
            opioid = paired(cells, model, arm, 'strong_opioid')
            if not severity:
                continue
            low, high = bootstrap_ci(severity, rng)
            rows[' '.join(arm)] = {
                'n_pairs': len(severity),
                'severity_diff': round(mean(severity), 3),
                'severity_ci95': [round(low, 3), round(high, 3)],
                'severity_p': round(sign_flip_p(severity, rng), 4),
                'strong_opioid_diff': round(mean(opioid), 3),
                'strong_opioid_p': round(sign_flip_p(opioid, rng), 4) if any(opioid) else 1.0,
            }
        out['models'][model] = rows
    return out


if __name__ == '__main__':
    import sys
    print(json.dumps(summarise(*load(sys.argv[1] if len(sys.argv) > 1 else '')), indent=1))
