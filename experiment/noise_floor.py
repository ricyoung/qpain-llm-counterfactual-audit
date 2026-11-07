"""Noise floor: how often a model's answer changes when nothing changes.

Same-condition flip rate: pairs of seeds for one (case, identity[, name]) whose
strong-opioid choice differs. Counterfactual flip rate: same case and seed, the
reference identity against another, whose choice differs. A counterfactual rate
at or below the floor is indistinguishable from sampling noise (after
Bellibatlu et al. 2026). Also reports severity disagreement rates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

from analyze import RAW, REFERENCE, legacy_kind, parse

KINDS = {'_v2': 'explicit (v2)', '_named': 'names (v3)'}


def kind_of(vignette: str) -> str | None:
    return next((k for k in ('_named', '_es', '_pt', '_ja', '_v2') if vignette.endswith(k)), None)


def main() -> dict:
    cells = defaultdict(dict)        # (kind, model) -> {(vignette, race, sex, slot, seed): parsed}
    for path in RAW.glob('*/*.json'):
        r = json.loads(path.read_text())
        kind = legacy_kind(r)
        if kind not in KINDS:
            continue
        x = parse(r['response'])
        if x is None:
            continue
        seed, slot = (str(r['rep']).split('-') + ['0'])[:2]
        cells[(kind, r['model'])][(r['vignette'], r['race'], r['sex'], slot, seed)] = x
    out = {}
    for (kind, model), table in sorted(cells.items()):
        groups = defaultdict(list)
        for (v, race, sex, slot, seed), x in table.items():
            groups[(v, race, sex, slot)].append(x)
        same = [(a['strong_opioid'] != b['strong_opioid'], a['severity'] != b['severity'])
                for xs in groups.values() for a, b in combinations(xs, 2)]
        cross = []
        for (v, race, sex, slot, seed), x in table.items():
            if (race, sex) == REFERENCE:
                continue
            ref = table.get((v, *REFERENCE, slot, seed))
            if ref:
                cross.append((x['strong_opioid'] != ref['strong_opioid'], x['severity'] != ref['severity']))
        rate = lambda pairs, i: round(sum(p[i] for p in pairs) / len(pairs), 3)
        out[f'{KINDS[kind]} | {model}'] = {
            'opioid_floor': rate(same, 0), 'opioid_counterfactual': rate(cross, 0),
            'severity_floor': rate(same, 1), 'severity_counterfactual': rate(cross, 1),
            'pairs_floor': len(same), 'pairs_counterfactual': len(cross)}
    return out


if __name__ == '__main__':
    for key, row in main().items():
        print(f"{key:32} opioid flips: noise {row['opioid_floor']:.3f} vs identity {row['opioid_counterfactual']:.3f}"
              f" | severity: noise {row['severity_floor']:.3f} vs identity {row['severity_counterfactual']:.3f}"
              f" | pairs {row['pairs_floor']}/{row['pairs_counterfactual']}")
