"""Do invalid answers depend on the identity in the prompt?  (review item S11)

An invalid answer is dropped from every paired comparison, so a model that failed more often for some
identities could hide or create a disparity among the answers that remain. For every configuration, case
set and cue with at least one invalid answer, this counts the invalid answers by identity and tests
whether they are uneven, treating a failure as an outcome of its own.

Invalid answers cluster by case (a hard case makes a reasoning model run out of tokens for every
identity), so the test permutes the failure flags within each case-and-seed block: a case that fails
for all eight identities cannot look like an identity effect. Statistic: chi-square of the failures per
identity against an even split; 20,000 permutations; Holm correction over the cells tested.

    python3.10 failure_by_identity.py        ->  outputs/failure_stats.json
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / 'outputs' / 'failure_stats.json'
B = 20_000


def identity_failure_test(flags, draws: int = B) -> dict:
    """flags: (n_blocks, 8) 0/1, one row per case-and-seed block, one column per identity."""
    flags = np.asarray(flags, dtype=int)
    n, k = flags.shape
    by_arm = flags.sum(axis=0)
    total = int(flags.sum())
    result = {'records': int(flags.size), 'invalid': total, 'by_arm': [int(x) for x in by_arm]}
    if total == 0:
        return dict(result, chi2=0.0, p=1.0)
    expected = total / k
    chi2 = float(((by_arm - expected) ** 2).sum() / expected)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(flags.tobytes()).digest()[:8], 'little'))
    hits = 0
    for start in range(0, draws, 2000):
        m = min(2000, draws - start)
        shuffled = rng.permuted(np.broadcast_to(flags, (m, n, k)).copy(), axis=2)     # within each block
        stats = ((shuffled.sum(axis=1) - expected) ** 2).sum(axis=1) / expected
        hits += int((stats >= chi2 - 1e-12).sum())
    return dict(result, chi2=round(chi2, 4), p=round((hits + 1) / (draws + 1), 5))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = round(running, 5)
    return adjusted


def collect() -> dict:
    """cells[(model, think, set, cue)][(case, slot, seed)][(race, sex)] = 1 if the answer was invalid."""
    import qpain
    from analyze import RAW, condition_of, parse as parse_main
    cells = defaultdict(lambda: defaultdict(dict))
    for path in RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        c = condition_of(record)
        if c['vset'] != 'v2' or c['lang'] != 'en':
            continue
        seed, slot = (str(record['rep']).split('-') + ['0'])[:2]
        key = (record['vignette'].split('_v2')[0], slot, seed)
        cells[(record['model'], c['think'], 'authored', c['cue'])][key][(record['race'], record['sex'])] = \
            int(parse_main(record['response']) is None)
    for path in qpain.RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        key = (record['item'], '0', str(record['seed']))
        cells[(record['model'], record.get('think', 'off'), 'qpain', record['cue'])][key][(record['race'], record['sex'])] = \
            int(qpain.parse(record['response']) is None)
    return cells


def main() -> dict:
    from run import ARMS
    rows = []
    for (model, think, vset, cue), blocks in sorted(collect().items()):
        full = [b for b in blocks.values() if all(a in b for a in ARMS)]              # complete blocks only
        flags = np.array([[b[a] for a in ARMS] for b in full], dtype=int)
        if not flags.size or not flags.sum():
            continue
        rows.append(dict(identity_failure_test(flags), label=f'{model} [{think}]', vset=vset, cue=cue, blocks=len(full)))
    for row, adjusted in zip(rows, holm([r['p'] for r in rows])):
        row['p_holm'] = adjusted
    result = {'identities': [' '.join(a) for a in ARMS], 'permutations': B, 'cells': rows,
              'total_invalid': sum(r['invalid'] for r in rows)}
    OUT.write_text(json.dumps(result, indent=1))
    for r in rows:
        print(f"{r['label']:40} {r['vset']:8} {r['cue']:8} invalid {r['invalid']:3} of {r['records']:5}  by identity {r['by_arm']}  p {r['p']}  Holm {r['p_holm']}")
    print(OUT)
    return result


if __name__ == '__main__':
    main()
