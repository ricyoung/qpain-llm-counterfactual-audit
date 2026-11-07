"""Panel analysis: every model configuration (local Ollama and OpenRouter), both case
sets, both cue types, read from the saved responses.

A configuration is a model with one thinking mode. For each configuration, case set
(the six authored v2 cases; the 50 Q-Pain cases) and cue (explicit labels; names):

  baseline     reference-arm (White man) means of each outcome
  identity     each of the 7 other identities minus the reference, paired on case, seed
               and name slot. The case is the resampling unit: a cluster bootstrap over
               cases gives the 95% CI, and a sign-flip test on the per-case mean
               differences gives p_case (exact when there are 16 cases or fewer, else
               from 200,000 random sign patterns). Each comparison draws from its own
               seed, a hash of its data, so no result depends on which other comparisons
               were computed first. The pair-level p_pair of the main study is kept
               alongside for comparison (10,000 patterns).
               Holm over the 7 identities within configuration x set x cue x outcome.
  equivalence  the 90% cluster CI inside +/-1 severity point or +/-0.10 proportion (TOST)
  contrasts    each race minus White within sex, and woman minus man within race
  flips        how often a decision changes: same prompt on another seed (noise floor),
               identity vs reference on the same seed and on other seeds
  validity     invalid responses counted per cell, never dropped silently

Across configurations:
  thinking     the same weights with reasoning on vs off (or effort high vs low): the
               change in each identity difference, per case, bootstrapped over cases
  precision    gemma3 12B 4-bit (local) vs gemma-3-12b bf16 (OpenRouter), same contrast

With six authored cases the smallest attainable two-sided p_case is 2/64 = 0.031, so
no authored-set identity can pass Holm at 0.05 on the case-level test; the 50 Q-Pain
cases carry the inferential weight.

    python3.10 analyze_panel.py             ->  outputs/panel_stats.json
    python3.10 analyze_panel.py --report    (also prints a plain-text summary)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import qpain
from analyze import condition_of, parse as parse_main
from run import ARMS, RAW, REFERENCE

HERE = Path(__file__).parent
OUT = HERE / 'outputs' / 'panel_stats.json'
B = 10_000        # bootstrap resamples; also the sign patterns for the pair-level p
B_FLIP = 200_000  # random sign patterns for a case-level p over more than 16 cases
FLIP_CHUNK = 20_000
SEED = 13
MIN_CASES = 3   # a cluster bootstrap over fewer cases is degenerate; report nothing
MARGINS = {'severity': 1.0, 'strong_opioid': 0.10, 'offer': 0.10, 'high_dose': 0.10}
OUTCOMES = {'authored': ('severity', 'strong_opioid'), 'qpain': ('severity', 'offer', 'high_dose')}
OTHERS = [a for a in ARMS if a != REFERENCE]
RACES = sorted({race for race, _ in ARMS} - {REFERENCE[0]})
SEXES = sorted({sex for _, sex in ARMS})
MODE_ORDER = ['off', 'low', 'medium', 'on', 'high']
PRECISION_PAIR = (('gemma3:12b', 'off'), ('google/gemma-3-12b-it', 'off'))


# ---------------------------------------------------------------- loading

def _usage(record: dict) -> dict:
    usage = record.get('usage') or {}
    details = usage.get('completion_tokens_details') or {}
    tokens = usage.get('completion_tokens', record.get('eval_count'))   # absent in the earliest local records
    return {'cost': float(record.get('cost') or 0.0),
            'completion_tokens': tokens or 0, 'token_records': 1 if tokens else 0,
            'reasoning_tokens': details.get('reasoning_tokens') or 0,
            'thinking_chars': len(record.get('thinking') or ''),
            'seconds': float(record.get('seconds') or 0.0)}


def load() -> tuple[dict, dict, dict]:
    """cells[(model, think, set, cue)][(case, slot, seed, race, sex)] = parsed outcomes,
    plus per-cell record counts and per-configuration usage totals."""
    cells, counts = defaultdict(dict), defaultdict(lambda: {'records': 0, 'invalid': 0})
    usage = defaultdict(lambda: defaultdict(float))

    def add(cell, key, parsed, record):
        counts[cell]['records'] += 1
        for name, value in _usage(record).items():
            usage[cell[:2]][name] += value
        usage[cell[:2]]['records'] += 1
        if parsed is None:
            counts[cell]['invalid'] += 1
        else:
            cells[cell][key] = parsed

    for path in RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        c = condition_of(record)
        if c['vset'] != 'v2' or c['lang'] != 'en':
            continue
        seed, slot = (str(record['rep']).split('-') + ['0'])[:2]
        add((record['model'], c['think'], 'authored', c['cue']),
            (record['vignette'].split('_v2')[0], slot, seed, record['race'], record['sex']),
            parse_main(record['response']), record)
    for path in qpain.RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        add((record['model'], record.get('think', 'off'), 'qpain', record['cue']),
            (record['item'], '0', str(record['seed']), record['race'], record['sex']),
            qpain.parse(record['response']), record)
    return cells, counts, usage


def by_arm(cells: dict) -> dict:
    out = defaultdict(dict)
    for (case, slot, seed, race, sex), parsed in cells.items():
        out[(race, sex)][(case, slot, seed)] = parsed
    return out


# ---------------------------------------------------------------- statistics

def differences(arms: dict, pairs: list[tuple], field: str) -> dict[str, list[float]]:
    """Per case: outcome(a) - outcome(b) over the (a, b) arm pairs, matched on case, slot, seed."""
    out = defaultdict(list)
    for a, b in pairs:
        reference = arms.get(b, {})
        for key, x in arms.get(a, {}).items():
            y = reference.get(key)
            if y is not None:
                out[key[0]].append(x[field] - y[field])
    return out


def sign_flip_p(values, rng, draws: int = B) -> float:
    v = np.asarray(values, dtype=float)
    observed = abs(v.mean())
    if observed < 1e-12:
        return 1.0
    if len(v) <= 16:                                   # exact over all 2^n sign patterns
        signs = ((np.arange(2 ** len(v))[:, None] >> np.arange(len(v))) & 1) * 2 - 1
        return float(np.mean(np.abs(signs @ v / len(v)) >= observed - 1e-12))
    hits = 0
    for start in range(0, draws, FLIP_CHUNK):          # in chunks, to bound memory
        signs = rng.integers(0, 2, size=(min(FLIP_CHUNK, draws - start), len(v)), dtype=np.int8) * 2 - 1
        hits += int(np.sum(np.abs(signs @ v / len(v)) >= observed - 1e-12))
    return float((hits + 1) / (draws + 1))


def comparison_rng(*arrays) -> np.random.Generator:
    """A generator seeded from one comparison's own data. Its interval and p-value then do not
    depend on which other comparisons were computed before it, or exist at all."""
    digest = hashlib.sha256()
    for values in arrays:
        digest.update((np.round(np.asarray(values, dtype=float), 9) + 0.0).tobytes())   # + 0.0: no -0.0
    return np.random.default_rng([SEED, int.from_bytes(digest.digest()[:8], 'little')])


def infer(per_case: dict[str, list[float]], margin: float) -> dict | None:
    if len(per_case) < MIN_CASES:
        return None
    case_means = np.array([np.mean(per_case[c]) for c in sorted(per_case)])
    pairs = np.concatenate([per_case[c] for c in sorted(per_case)])
    rng = comparison_rng(case_means, pairs)
    boot = rng.choice(case_means, size=(B, len(case_means))).mean(axis=1)
    lo95, hi95, lo90, hi90 = np.percentile(boot, [2.5, 97.5, 5, 95])
    return {'estimate': round(float(case_means.mean()), 4),
            'ci95': [round(float(lo95), 4), round(float(hi95), 4)],
            'ci90': [round(float(lo90), 4), round(float(hi90), 4)],
            'equivalent': bool(-margin < lo90 and hi90 < margin),
            'p_case': round(sign_flip_p(case_means, rng, B_FLIP), 6),
            'p_pair': round(sign_flip_p(pairs, rng), 6),
            'n_cases': len(case_means), 'n_pairs': len(pairs)}


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = round(running, 5)
    return adjusted


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=pvalues.__getitem__, reverse=True)
    adjusted, running = [0.0] * n, 1.0
    for rank, i in enumerate(order):
        running = min(running, pvalues[i] * n / (n - rank))
        adjusted[i] = round(running, 5)
    return adjusted


def flip_rates(cells: dict, fields: tuple) -> dict:
    """Same-prompt floor vs identity flips on the same and on other seeds."""
    prompts = defaultdict(dict)
    for (case, slot, seed, race, sex), x in cells.items():
        prompts[(case, slot, race, sex)][seed] = x
    out = {}
    for field in fields:
        floor, same, cross = [], [], []
        for (case, slot, race, sex), seeds in prompts.items():
            floor += [a[field] != b[field] for a, b in combinations(seeds.values(), 2)]
            if (race, sex) == REFERENCE:
                continue
            for s, x in seeds.items():
                for t, r in prompts.get((case, slot, *REFERENCE), {}).items():
                    (same if s == t else cross).append(x[field] != r[field])
        out[field] = {name: round(float(np.mean(v)), 4) if v else None
                      for name, v in (('floor', floor), ('identity_same_seed', same), ('identity_cross_seed', cross))}
    return out


def eta_squared(cells: dict) -> float | None:
    groups = defaultdict(list)
    for (case, *_), x in cells.items():
        groups[case].append(x['severity'])
    values = np.array([v for g in groups.values() for v in g], dtype=float)
    total = float(((values - values.mean()) ** 2).sum())
    between = sum(len(g) * (np.mean(g) - values.mean()) ** 2 for g in groups.values())
    return round(float(between / total), 4) if total else None


def reference_agreement(cells: dict) -> dict:
    """Q-Pain only: agreement with the clinical expert's reference answer, which is to
    offer the opioid in all 50 open cases, at the dose or supply stated for each case."""
    dose = {item['id']: 'high' if item['reference_dosage'].strip().lower().startswith('high') else 'low'
            for item in qpain.load_vignettes()}
    offered = [x for (case, *_), x in cells.items() if x['offer']]
    match = [x['high_dose'] == (dose[case] == 'high') for (case, *_), x in cells.items() if x['offer']]
    return {'offer_rate': round(float(np.mean([x['offer'] for x in cells.values()])), 4),
            'dose_match_when_offered': round(float(np.mean(match)), 4) if match else None,
            'full_match': round(float(sum(match) / len(cells)), 4),
            'reference_high_dose_share': round(float(np.mean([v == 'high' for v in dose.values()])), 4),
            'n_offered': len(offered)}


# ---------------------------------------------------------------- per configuration

def analyse_cell(cells: dict, fields: tuple) -> dict:
    arms = by_arm(cells)
    reference = arms.get(REFERENCE, {})
    out = {'baseline': {f: round(float(np.mean([x[f] for x in reference.values()])), 4) if reference else None
                        for f in fields},
           'overall': {f: round(float(np.mean([x[f] for x in cells.values()])), 4) for f in fields},
           'eta_squared_case': eta_squared(cells), 'flips': flip_rates(cells, fields),
           'identity': {}, 'race': {}, 'sex': {}}
    if 'offer' in fields:
        out['reference_agreement'] = reference_agreement(cells)
    for field in fields:
        rows = {' '.join(arm): infer(differences(arms, [(arm, REFERENCE)], field), MARGINS[field])
                for arm in OTHERS}
        rows = {k: v for k, v in rows.items() if v}
        for (name, row), adjusted in zip(rows.items(), holm([r['p_case'] for r in rows.values()])):
            row['p_case_holm'] = adjusted
        for (name, row), adjusted in zip(rows.items(), holm([r['p_pair'] for r in rows.values()])):
            row['p_pair_holm'] = adjusted
        out['identity'][field] = rows
        out['race'][field] = {race: infer(differences(arms, [((race, s), (REFERENCE[0], s)) for s in SEXES], field),
                                          MARGINS[field]) for race in RACES}
        out['sex'][field] = infer(differences(arms, [((race, 'woman'), (race, 'man')) for race in [REFERENCE[0], *RACES]],
                                              field), MARGINS[field])
    return out


def contrast(cells_a: dict, cells_b: dict, fields: tuple) -> dict:
    """Configuration B minus configuration A: the change in each identity difference and
    in the baseline, per case, bootstrapped over the cases both share."""
    arms_a, arms_b = by_arm(cells_a), by_arm(cells_b)
    out = {'identity': {}, 'baseline_shift': {}}
    for field in fields:
        rows = {}
        for arm in OTHERS:
            da = differences(arms_a, [(arm, REFERENCE)], field)
            db = differences(arms_b, [(arm, REFERENCE)], field)
            shared = sorted(set(da) & set(db))
            rows[' '.join(arm)] = infer({c: [np.mean(db[c]) - np.mean(da[c])] for c in shared}, MARGINS[field])
        rows = {k: v for k, v in rows.items() if v}
        for (name, row), adjusted in zip(rows.items(), holm([r['p_case'] for r in rows.values()])):
            row['p_case_holm'] = adjusted
        out['identity'][field] = rows
        shift = defaultdict(list)
        for arm in ARMS:
            for key, x in arms_b.get(arm, {}).items():
                shift[key[0]].append(x[field])
        base = defaultdict(list)
        for arm in ARMS:
            for key, x in arms_a.get(arm, {}).items():
                base[key[0]].append(x[field])
        shared = sorted(set(shift) & set(base))
        out['baseline_shift'][field] = infer({c: [np.mean(shift[c]) - np.mean(base[c])] for c in shared},
                                             MARGINS[field])
    return out


def cell_agreement(cells_a: dict, cells_b: dict, field: str) -> dict:
    """Correlation and mean absolute gap of prompt-level means (averaged over seeds)."""
    def means(cells):
        groups = defaultdict(list)
        for (case, slot, seed, race, sex), x in cells.items():
            groups[(case, slot, race, sex)].append(x[field])
        return {k: float(np.mean(v)) for k, v in groups.items()}
    a, b = means(cells_a), means(cells_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return {'n_prompts': len(shared)}
    x, y = np.array([a[k] for k in shared]), np.array([b[k] for k in shared])
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
    return {'n_prompts': len(shared), 'pearson_r': round(r, 4) if r is not None else None,
            'mean_abs_gap': round(float(np.abs(x - y).mean()), 4), 'mean_gap_b_minus_a': round(float((y - x).mean()), 4)}


def thinking_pairs(configs: set) -> list[tuple]:
    models = defaultdict(set)
    for model, think in configs:
        models[model].add(think)
    pairs = []
    for model, modes in sorted(models.items()):
        ordered = sorted(modes, key=MODE_ORDER.index)
        pairs += [(model, a, b) for a, b in combinations(ordered, 2)]
    return pairs


def main() -> dict:
    cells, counts, usage = load()
    stats = {'reference_arm': ' '.join(REFERENCE), 'margins': MARGINS, 'bootstrap': B,
             'configurations': {}, 'thinking': {}, 'precision': {}, 'study_wide': {}}
    for (model, think, vset, cue) in sorted(cells):
        label = f'{model} [{think}]'
        entry = stats['configurations'].setdefault(label, {'model': model, 'think': think, 'usage': {}, 'sets': {}})
        entry['sets'].setdefault(vset, {})[cue] = dict(
            analyse_cell(cells[(model, think, vset, cue)], OUTCOMES[vset]), **counts[(model, think, vset, cue)])
    for (model, think), totals in usage.items():
        label = f'{model} [{think}]'
        if label in stats['configurations']:
            n = totals['records'] or 1
            stats['configurations'][label]['usage'] = {
                'records': int(totals['records']), 'cost_usd': round(totals['cost'], 4),
                'mean_completion_tokens': (round(totals['completion_tokens'] / totals['token_records'], 1)
                                           if totals['token_records'] else None),
                'records_with_token_counts': int(totals['token_records']),
                'mean_reasoning_tokens': round(totals['reasoning_tokens'] / n, 1),
                'mean_thinking_chars': round(totals['thinking_chars'] / n, 1),
                'mean_seconds': round(totals['seconds'] / n, 2)}
    configs = {(m, t) for m, t, _, _ in cells}
    for model, a, b in thinking_pairs(configs):
        result = {}
        for vset, fields in OUTCOMES.items():
            for cue in ('explicit', 'names'):
                ca, cb = cells.get((model, a, vset, cue)), cells.get((model, b, vset, cue))
                if ca and cb:
                    result.setdefault(vset, {})[cue] = contrast(ca, cb, fields)
        if result:
            stats['thinking'][f'{model}: {b} minus {a}'] = result
    (ma, ta), (mb, tb) = PRECISION_PAIR
    for vset, fields in OUTCOMES.items():
        for cue in ('explicit', 'names'):
            ca, cb = cells.get((ma, ta, vset, cue)), cells.get((mb, tb, vset, cue))
            if ca and cb:
                stats['precision'].setdefault(vset, {})[cue] = dict(
                    contrast(ca, cb, fields), agreement={f: cell_agreement(ca, cb, f) for f in fields})
    stats['study_wide'] = study_wide(stats)
    OUT.write_text(json.dumps(stats, indent=1))
    return stats


def study_wide(stats: dict) -> dict:
    """Every identity comparison in every configuration, set, cue and outcome, BH-adjusted."""
    tests = []
    for label, entry in stats['configurations'].items():
        for vset, cues in entry['sets'].items():
            for cue, cell in cues.items():
                for field, rows in cell['identity'].items():
                    tests += [(label, vset, cue, field, arm, row['p_case'], row['estimate']) for arm, row in rows.items()]
    q = benjamini_hochberg([t[5] for t in tests]) if tests else []
    hits = [dict(zip(('configuration', 'set', 'cue', 'outcome', 'identity', 'p_case', 'estimate'), t), q=qv)
            for t, qv in zip(tests, q) if qv < 0.05]
    return {'tests': len(tests), 'p_below_05': sum(t[5] < 0.05 for t in tests),
            'expected_under_null': round(0.05 * len(tests), 1), 'bh_q_below_05': hits}


# ---------------------------------------------------------------- report

def report(stats: dict) -> None:
    for vset, fields in OUTCOMES.items():
        for cue in ('explicit', 'names'):
            print(f'\n== {vset} / {cue} ==')
            print(f"{'configuration':44} {'n':>5} {'inv':>4} {'eta2':>5} "
                  + ' '.join(f'{f[:8]:>8}' for f in fields) + '  largest |identity diff| (Holm p_case)  equiv')
            for label, entry in stats['configurations'].items():
                cell = entry['sets'].get(vset, {}).get(cue)
                if not cell:
                    continue
                base = ' '.join(f"{cell['baseline'][f] if cell['baseline'][f] is not None else float('nan'):8.3f}"
                                for f in fields)
                worst = []
                for f in fields:
                    rows = cell['identity'].get(f, {})
                    if rows:
                        arm, row = max(rows.items(), key=lambda kv: abs(kv[1]['estimate']))
                        worst.append(f"{f[:3]} {row['estimate']:+.3f} {arm.split()[0][:3]}{arm.split()[1][0]} ({row['p_case_holm']:.2f})")
                equiv = '/'.join(str(sum(r['equivalent'] for r in cell['identity'].get(f, {}).values())) for f in fields)
                print(f"{label:44} {cell['records']:5} {cell['invalid']:4} {cell['eta_squared_case'] or 0:5.2f} {base}  "
                      + '; '.join(worst) + f'  {equiv}')
    sw = stats['study_wide']
    print(f"\nstudy-wide: {sw['tests']} identity tests, {sw['p_below_05']} with p_case < 0.05 "
          f"({sw['expected_under_null']} expected by chance), {len(sw['bh_q_below_05'])} with BH q < 0.05")
    for hit in sw['bh_q_below_05']:
        print(f"  {hit['configuration']} {hit['set']}/{hit['cue']} {hit['outcome']} {hit['identity']}: "
              f"{hit['estimate']:+.3f}, p {hit['p_case']:.4f}, q {hit['q']:.4f}")
    for name, result in stats['thinking'].items():
        for vset, cues in result.items():
            for cue, c in cues.items():
                shifts = ', '.join(f"{f[:3]} {s['estimate']:+.3f}" for f, s in c['baseline_shift'].items() if s)
                sig = [f"{f[:3]} {arm} {row['estimate']:+.3f}" for f, rows in c['identity'].items()
                       for arm, row in rows.items() if row['p_case_holm'] < 0.05]
                print(f"thinking {name} {vset}/{cue}: baseline {shifts}; identity changes Holm<0.05: {sig or 'none'}")
    for vset, cues in stats['precision'].items():
        for cue, c in cues.items():
            agree = ', '.join(f"{f[:3]} r={a.get('pearson_r')} gap={a.get('mean_gap_b_minus_a')}"
                              for f, a in c['agreement'].items())
            print(f'precision bf16 minus Q4 {vset}/{cue}: {agree}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    result = main()
    if parser.parse_args().report:
        report(result)
    print(OUT)
