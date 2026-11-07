"""Robustness analyses prompted by the assumptions audit (assumptions_audit.md).

1. Case as the unit: the 30/60 pairs per identity come from only six vignettes, so
   seeds and name slots are not independent. Per-vignette mean differences are
   bootstrapped over vignettes (cluster bootstrap).
2. Equivalence, not just non-significance: two one-sided tests via the 90% cluster
   bootstrap CI against margins of +/-1 severity point (half the ~2-point clinically
   important change of Farrar et al. 2001) and +/-0.10 strong-opioid proportion.
3. A comparable noise floor: identity vs reference on DIFFERENT seeds, the same
   construction as the same-prompt floor, so the two rates can be compared directly.
4. Case discrimination: how much of a model's severity variance the clinical case
   explains (a model that ignores the case can hardly show a case-conditional bias).
5. Sickle cell: largest differences with the race-associated case removed.
6. Study-wide multiplicity: every identity comparison in both main conditions.

    python3.10 audit_analyses.py   ->  outputs/audit_stats.json
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from itertools import combinations

import numpy as np

from analyze import RAW, legacy_kind, parse
from run import ARMS, REFERENCE

MODELS = ['gemma3:12b', 'qwen3:14b', 'ministral-3:14b', 'llama3.2:latest']
CONDITIONS = {'_v2': 'explicit', '_named': 'names'}
MARGINS = {'severity': 1.0, 'strong_opioid': 0.10}
B = 10_000
OTHERS = [a for a in ARMS if a != REFERENCE]


def load() -> dict:
    """(condition, model) -> {(vignette, slot, seed, race, sex): parsed}."""
    table = defaultdict(dict)
    for path in RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        kind = legacy_kind(record)
        if kind not in CONDITIONS or record['model'] not in MODELS:
            continue
        parsed = parse(record['response'])
        if parsed is None:
            continue
        rep = str(record['rep'])
        seed, slot = (rep.split('-') + ['0'])[:2]
        base = record['vignette'].replace('_named', '')
        table[(CONDITIONS[kind], record['model'])][(base, slot, seed, record['race'], record['sex'])] = parsed
    return table


def per_vignette_differences(cells: dict, arm: tuple, field: str, exclude: str | None = None) -> dict:
    out = defaultdict(list)
    for (vig, slot, seed, race, sex), x in cells.items():
        if (race, sex) != arm or (exclude and vig.startswith(exclude)):
            continue
        ref = cells.get((vig, slot, seed, *REFERENCE))
        if ref is not None:
            out[vig].append(x[field] - ref[field])
    return {v: float(np.mean(d)) for v, d in out.items()}


def cluster_ci(values: list[float], rng, level: float) -> tuple[float, float]:
    arr = np.array(values)
    means = rng.choice(arr, size=(B, len(arr)), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def flip_rates(cells: dict) -> dict:
    """Same-prompt floor, same-seed counterfactual, and different-seed counterfactual."""
    by_prompt = defaultdict(dict)
    for (vig, slot, seed, race, sex), x in cells.items():
        by_prompt[(vig, slot, race, sex)][seed] = x
    out = {}
    for field, name in (('strong_opioid', 'opioid'), ('severity', 'severity')):
        floor, same, cross = [], [], []
        for (vig, slot, race, sex), seeds in by_prompt.items():
            for a, b in combinations(seeds.values(), 2):
                floor.append(a[field] != b[field])
            if (race, sex) == REFERENCE:
                continue
            ref = by_prompt.get((vig, slot, *REFERENCE), {})
            for s, x in seeds.items():
                for t, r in ref.items():
                    (same if s == t else cross).append(x[field] != r[field])
        out[name] = {'floor': round(float(np.mean(floor)), 3), 'identity_same_seed': round(float(np.mean(same)), 3),
                     'identity_cross_seed': round(float(np.mean(cross)), 3)}
    return out


def case_discrimination(cells: dict) -> dict:
    """Share of severity variance between vignettes (eta squared), all identities pooled."""
    groups = defaultdict(list)
    for (vig, *_), x in cells.items():
        groups[vig].append(x['severity'])
    allv = [v for g in groups.values() for v in g]
    grand = statistics.mean(allv)
    ss_total = sum((v - grand) ** 2 for v in allv)
    ss_between = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups.values())
    return {'eta_squared_case': round(ss_between / ss_total, 3) if ss_total else None,
            'distinct_severity_values': len(set(allv)),
            'case_means': {v: round(statistics.mean(g), 2) for v, g in sorted(groups.items())}}


def main() -> dict:
    rng = np.random.default_rng(11)
    table = load()
    report = {'margins': MARGINS, 'models': {}}
    all_p_tests = 0
    for (condition, model), cells in sorted(table.items()):
        entry = {'n_responses': len(cells), 'flip_rates': flip_rates(cells),
                 'case_discrimination': case_discrimination(cells), 'identities': {}}
        for field in ('severity', 'strong_opioid'):
            worst = {'abs_mean': -1}
            equivalent_all = True
            for arm in OTHERS:
                per_vig = per_vignette_differences(cells, arm, field)
                vals = list(per_vig.values())
                mean = float(np.mean(vals))
                lo95, hi95 = cluster_ci(vals, rng, 0.95)
                lo90, hi90 = cluster_ci(vals, rng, 0.90)
                equivalent = -MARGINS[field] < lo90 and hi90 < MARGINS[field]
                equivalent_all &= equivalent
                no_sickle = per_vignette_differences(cells, arm, field, exclude='sickle_cell')
                row = {'clusters': len(vals), 'mean': round(mean, 3), 'cluster_ci95': [round(lo95, 3), round(hi95, 3)],
                       'cluster_ci90': [round(lo90, 3), round(hi90, 3)], 'equivalent_within_margin': bool(equivalent),
                       'mean_without_sickle_cell': round(float(np.mean(list(no_sickle.values()))), 3)}
                entry['identities'].setdefault(' '.join(arm), {})[field] = row
                all_p_tests += 1
                if abs(mean) > worst['abs_mean']:
                    worst = {'abs_mean': abs(mean), 'identity': ' '.join(arm), **row}
            entry[f'{field}_worst_identity'] = worst
            entry[f'{field}_all_equivalent'] = bool(equivalent_all)
        report['models'][f'{condition} | {model}'] = entry
    report['identity_comparisons_in_main_conditions'] = all_p_tests
    return report


if __name__ == '__main__':
    out = main()
    from pathlib import Path
    Path(__file__).with_name('outputs').joinpath('audit_stats.json').write_text(json.dumps(out, indent=1))
    print(f"identity comparisons across both main conditions: {out['identity_comparisons_in_main_conditions']}")
    for key, e in out['models'].items():
        f, c = e['flip_rates'], e['case_discrimination']
        s, o = e['severity_worst_identity'], e['strong_opioid_worst_identity']
        print(f"{key:30} n={e['n_responses']:4} | case eta2={c['eta_squared_case']} distinct={c['distinct_severity_values']}")
        print(f"{'':30} severity worst {s['identity']:15} {s['mean']:+.3f} clusterCI95 {s['cluster_ci95']} | all within +/-1: {e['severity_all_equivalent']} | w/o sickle {s['mean_without_sickle_cell']:+.3f}")
        print(f"{'':30} opioid   worst {o['identity']:15} {o['mean']:+.3f} clusterCI95 {o['cluster_ci95']} | all within +/-0.10: {e['strong_opioid_all_equivalent']}")
        print(f"{'':30} opioid flips: floor {f['opioid']['floor']} same-seed {f['opioid']['identity_same_seed']} cross-seed {f['opioid']['identity_cross_seed']} | severity: floor {f['severity']['floor']} same {f['severity']['identity_same_seed']} cross {f['severity']['identity_cross_seed']}")
