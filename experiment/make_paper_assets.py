"""Build every statistic, table and figure the manuscript reports, from the raw responses.

Outputs (all regenerated on each run):
  outputs/paper_stats.json      every number quoted in the paper
  ../paper/tables/*.tex         booktabs tables, \\input by main.tex
  ../paper/figures/*.pdf        Figure 1 (paired differences), Figure 2 (noise floor)

    python3.10 make_paper_assets.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze import RAW, legacy_kind, parse
from run import ARMS, HERE, NAMES, REFERENCE

PAPER = HERE.parent / 'paper'
TABLES, FIGURES = PAPER / 'tables', PAPER / 'figures'
ITERATIONS = 10_000
SEED = 7

MODELS = [  # display order; tag -> (short name, developer, parameters, licence)
    ('gemma3:12b', 'gemma3 12B', 'Google', '12.2B', 'Gemma Terms of Use'),
    ('qwen3:14b', 'qwen3 14B', 'Alibaba', '14.8B', 'Apache 2.0'),
    ('ministral-3:14b', 'ministral-3 14B', 'Mistral AI', '13.9B', 'Apache 2.0'),
    ('llama3.2:latest', 'llama3.2 3B', 'Meta', '3.2B', 'Llama 3.2 Community'),
]
SHORT = {tag: short for tag, short, *_ in MODELS}
CONDITIONS = [('', 'Pilot (self-rated)'), ('_v2', 'Explicit labels'), ('_named', 'Name cues')]
ARM_LABEL = {arm: f'{arm[0]} {arm[1]}' for arm in ARMS}
COMPARISON_ARMS = [arm for arm in ARMS if arm != REFERENCE]

# Palette: dataviz reference instance (validated categorical slots 1-2, muted ink, chrome).
BLUE, ORANGE, MUTED, GRID, AXIS, INK, INK2 = '#2a78d6', '#eb6834', '#898781', '#e1e0d9', '#c3c2b7', '#0b0b0b', '#52514e'


def kind_of(vignette: str) -> str:
    return next((k for k in ('_named', '_es', '_pt', '_ja', '_v2') if vignette.endswith(k)), '')


def load_records() -> list[dict]:
    records = []
    for path in RAW.glob('*/*.json'):
        record = json.loads(path.read_text())
        record['kind'] = legacy_kind(record)
        record['parsed'] = parse(record['response'])
        records.append(record)
    return records


def stated_ratings() -> dict[str, int]:
    """The self-reported rating embedded in each pilot (v1) vignette."""
    out = {}
    for item in json.loads((HERE / 'vignettes.json').read_text()):
        found = re.search(r'Rates pain (\d+)/10', item['case'])
        out[item['id']] = int(found.group(1)) if found else None
    return out


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running, m = [0.0] * len(pvalues), 0.0, len(pvalues)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(ITERATIONS, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(values.mean())
    if observed == 0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(ITERATIONS, len(values)))
    extreme = int((np.abs((signs * values).mean(axis=1)) >= observed - 1e-12).sum())
    return (extreme + 1) / (ITERATIONS + 1)


def cells_for(records: list[dict], kind: str, model: str) -> dict:
    """(vignette, rep, race, sex) -> parsed, for one condition and model."""
    return {(r['vignette'], str(r['rep']), r['race'], r['sex']): r['parsed']
            for r in records if r['kind'] == kind and r['model'] == model and r['parsed']}


def paired_differences(cells: dict, arm: tuple, field: str) -> np.ndarray:
    values = []
    for (vignette, rep, race, sex), parsed in cells.items():
        if (race, sex) != arm:
            continue
        reference = cells.get((vignette, rep, *REFERENCE))
        if reference is not None:
            values.append(parsed[field] - reference[field])
    return np.array(values, dtype=float)


def flip_rates(cells: dict) -> dict:
    """Same-prompt (floor) and same-seed counterfactual disagreement rates."""
    groups = defaultdict(list)
    for (vignette, rep, race, sex), parsed in cells.items():
        slot = rep.split('-')[1] if '-' in rep else '0'
        groups[(vignette, race, sex, slot)].append(parsed)
    floor = [(a['strong_opioid'] != b['strong_opioid'], a['severity'] != b['severity'])
             for group in groups.values() for a, b in combinations(group, 2)]
    counterfactual = []
    for (vignette, rep, race, sex), parsed in cells.items():
        if (race, sex) == REFERENCE:
            continue
        reference = cells.get((vignette, rep, *REFERENCE))
        if reference is not None:
            counterfactual.append((parsed['strong_opioid'] != reference['strong_opioid'],
                                   parsed['severity'] != reference['severity']))
    rate = lambda pairs, i: round(sum(p[i] for p in pairs) / len(pairs), 3) if pairs else None
    return {'opioid_floor': rate(floor, 0), 'opioid_counterfactual': rate(counterfactual, 0),
            'severity_floor': rate(floor, 1), 'severity_counterfactual': rate(counterfactual, 1),
            'floor_pairs': len(floor), 'counterfactual_pairs': len(counterfactual)}


def minimum_detectable(differences: list[np.ndarray]) -> dict:
    """Two-sided alpha .05, power .80: 2.80 x SD / sqrt(n), SD pooled over the seven arms."""
    pooled = np.concatenate(differences)
    n = int(np.mean([len(d) for d in differences]))
    sd = float(pooled.std(ddof=1)) if len(pooled) > 1 else 0.0
    return {'sd_paired': round(sd, 3), 'n_pairs': n, 'mde': round(2.80 * sd / np.sqrt(n), 3) if n else None}


def compute() -> dict:
    rng = np.random.default_rng(SEED)
    records = load_records()
    stated = stated_ratings()
    stats = {'conditions': {}, 'per_name': {}, 'totals': {}}
    for kind, label in CONDITIONS:
        condition = {'label': label, 'models': {}}
        for tag, *_ in MODELS:
            mine = [r for r in records if r['kind'] == kind and r['model'] == tag]
            parsed = [r['parsed'] for r in mine if r['parsed']]
            if not mine:
                continue
            cells = cells_for(records, kind, tag)
            severities = [p['severity'] for p in parsed]
            entry = {
                'responses': len(mine), 'parsed': len(parsed),
                'mean_severity': round(statistics.mean(severities), 2),
                'sd_severity': round(statistics.pstdev(severities), 2),
                'min_severity': min(severities), 'max_severity': max(severities),
                'strong_opioid_rate': round(statistics.mean(p['strong_opioid'] for p in parsed), 3),
                'arms': {}, 'flips': flip_rates(cells),
            }
            if kind == '':
                anchored = [r['parsed']['severity'] == stated[r['vignette']] for r in mine if r['parsed']]
                entry['anchored_to_stated_rating'] = round(statistics.mean(anchored), 3)
            per_measure = {'severity': [], 'strong_opioid': []}
            for arm in COMPARISON_ARMS:
                row = {}
                for field in ('severity', 'strong_opioid'):
                    diffs = paired_differences(cells, arm, field)
                    per_measure[field].append(diffs)
                    low, high = bootstrap_ci(diffs, rng)
                    row[field] = {'n_pairs': int(len(diffs)), 'diff': round(float(diffs.mean()), 3),
                                  'ci95': [round(low, 3), round(high, 3)],
                                  'p': round(sign_flip_p(diffs, rng), 4)}
                entry['arms'][ARM_LABEL[arm]] = row
            for field in ('severity', 'strong_opioid'):
                adjusted = holm([entry['arms'][ARM_LABEL[a]][field]['p'] for a in COMPARISON_ARMS])
                for arm, adj in zip(COMPARISON_ARMS, adjusted):
                    entry['arms'][ARM_LABEL[arm]][field]['p_holm'] = round(adj, 4)
                entry[f'mde_{field}'] = minimum_detectable(per_measure[field])
            condition['models'][tag] = entry
        stats['conditions'][kind] = condition
    # Per-name means for the name-cue condition.
    for tag, *_ in MODELS:
        by_name = defaultdict(list)
        for r in records:
            if r['kind'] == '_named' and r['model'] == tag and r['parsed']:
                by_name[r['name']].append(r['parsed'])
        stats['per_name'][tag] = {
            name: {'n': len(v), 'mean_severity': round(statistics.mean(p['severity'] for p in v), 3),
                   'strong_opioid_rate': round(statistics.mean(p['strong_opioid'] for p in v), 3)}
            for name, v in by_name.items()}
    kinds = defaultdict(int)
    for r in records:
        kinds[r['kind']] += 1
    stats['totals'] = {'responses_by_condition': dict(kinds),
                       'main_conditions': kinds['_v2'] + kinds['_named'],
                       'all_english': kinds[''] + kinds['_v2'] + kinds['_named']}
    return stats


# ---------------------------------------------------------------- tables

def fmt_p(p: float) -> str:
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def table_models() -> str:
    rows = [f"{short} & \\texttt{{{tag}}} & {dev} & {params} & Q4\\_K\\_M & {lic} \\\\"
            for tag, short, dev, params, lic in MODELS]
    return '\n'.join([
        r'\begin{tabular}{llllll}', r'\toprule',
        r'Model & Ollama tag & Developer & Parameters & Quantisation & Licence \\', r'\midrule',
        *rows, r'\bottomrule', r'\end{tabular}'])


def table_baselines(stats: dict) -> str:
    lines = [r'\begin{tabular}{llrrrcr}', r'\toprule',
             r'Condition & Model & Responses & Parsed & Mean severity (SD) & Range & Strong-opioid rate \\', r'\midrule']
    for kind, label in CONDITIONS:
        for i, (tag, *_) in enumerate(MODELS):
            e = stats['conditions'][kind]['models'].get(tag)
            if not e:
                continue
            lines.append(f"{label if i == 0 else ''} & {SHORT[tag]} & {e['responses']:,} & {e['parsed']:,} & "
                         f"{e['mean_severity']:.2f} ({e['sd_severity']:.2f}) & {e['min_severity']}--{e['max_severity']} & "
                         f"{e['strong_opioid_rate']:.2f} \\\\")
        lines.append(r'\addlinespace')
    lines[-1:] = [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def table_differences(stats: dict) -> str:
    lines = [r'\begin{tabular}{lllrrlcrr}', r'\toprule',
             r'Condition & Model & Outcome & $n$/arm & Largest $\Delta$ & Identity & 95\% CI & min $p$ & min $p_{\mathrm{Holm}}$ \\',
             r'\midrule']
    for kind, label in CONDITIONS[1:]:
        for i, (tag, *_) in enumerate(MODELS):
            e = stats['conditions'][kind]['models'][tag]
            n = next(iter(e['arms'].values()))['severity']['n_pairs']
            for j, (field, outcome) in enumerate([('severity', 'Severity'), ('strong_opioid', 'Strong opioid')]):
                arms = e['arms']
                worst = max(arms, key=lambda a: abs(arms[a][field]['diff']))
                d = arms[worst][field]
                digits = 2 if field == 'severity' else 3
                if all(arms[a][field]['diff'] == 0 for a in arms):
                    delta, identity, ci = '0', 'all identities', '--'
                else:
                    delta, identity = f"{d['diff']:+.{digits}f}", worst
                    ci = f"[{d['ci95'][0]:.{digits}f}, {d['ci95'][1]:.{digits}f}]"
                lines.append(f"{label if i == 0 and j == 0 else ''} & {SHORT[tag] if j == 0 else ''} & {outcome} & {n} & "
                             f"{delta} & {identity} & {ci} & {fmt_p(min(arms[a][field]['p'] for a in arms))} & "
                             f"{fmt_p(min(arms[a][field]['p_holm'] for a in arms))} \\\\")
        lines.append(r'\addlinespace')
    lines[-1:] = [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def table_noise(stats: dict) -> str:
    lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r' & & \multicolumn{2}{c}{Strong-opioid flip rate} & \multicolumn{2}{c}{Severity disagreement rate} & '
             r'\multicolumn{2}{c}{Minimum detectable $\Delta$} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'Condition & Model & Floor & Identity & Floor & Identity & Severity & Opioid \\', r'\midrule']
    for kind, label in CONDITIONS[1:]:
        for i, (tag, *_) in enumerate(MODELS):
            e = stats['conditions'][kind]['models'][tag]
            f = e['flips']
            lines.append(f"{label if i == 0 else ''} & {SHORT[tag]} & {f['opioid_floor']:.3f} & {f['opioid_counterfactual']:.3f} & "
                         f"{f['severity_floor']:.3f} & {f['severity_counterfactual']:.3f} & "
                         f"{e['mde_severity']['mde']:.2f} & {e['mde_strong_opioid']['mde']:.3f} \\\\")
        lines.append(r'\addlinespace')
    lines[-1:] = [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def table_names(stats: dict) -> str:
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Identity & Name & ' + ' & '.join(SHORT[t] for t, *_ in MODELS) + r' \\', r'\midrule']
    for arm in ARMS:
        for j, name in enumerate(NAMES[arm]):
            values = ' & '.join(f"{stats['per_name'][t][name]['mean_severity']:.2f}" for t, *_ in MODELS)
            lines.append(f"{ARM_LABEL[arm] if j == 0 else ''} & {name} & {values} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# ---------------------------------------------------------------- figures

def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_differences(stats: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'text.color': INK,
                         'axes.labelcolor': INK2, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(6.5, 7.6))
    series = [('_v2', 'Explicit labels', BLUE, 'o'), ('_named', 'Name cues', ORANGE, 'D')]
    labels = [ARM_LABEL[a] for a in COMPARISON_ARMS]
    y = np.arange(len(labels))[::-1]
    for row, (tag, *_) in enumerate(MODELS):
        for col, (field, title, limit) in enumerate([('severity', 'Pain severity (0--10 points)', 0.35),
                                                     ('strong_opioid', 'Strong opioid (proportion)', 0.25)]):
            ax = axes[row, col]
            style(ax)
            ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
            for k, (kind, name, colour, marker) in enumerate(series):
                offset = 0.18 if k == 0 else -0.18
                arms = stats['conditions'][kind]['models'][tag]['arms']
                d = np.array([arms[l][field]['diff'] for l in labels])
                lo = np.array([arms[l][field]['ci95'][0] for l in labels])
                hi = np.array([arms[l][field]['ci95'][1] for l in labels])
                ax.hlines(y + offset, lo, hi, color=colour, linewidth=1.5, zorder=2)
                ax.plot(d, y + offset, linestyle='none', marker=marker, markersize=5.5, color=colour,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=3, label=name if (row, col) == (0, 0) else None)
            ax.set_xlim(-limit, limit)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.tick_params(axis='y', labelleft=(col == 0))
            ax.set_ylim(-0.7, len(labels) - 0.3)
            if row == 0:
                ax.set_title(title.replace('--', '–'), fontsize=8.5, color=INK, loc='left', pad=6)
            if col == 0:
                ax.set_ylabel(SHORT[tag], fontsize=8, color=INK2, labelpad=8)
    fig.supxlabel('Paired difference, identity minus White man reference (mean and bootstrap 95% CI)',
                  fontsize=8, color=INK2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path)
    plt.close(fig)


def figure_noise(stats: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'text.color': INK,
                         'axes.labelcolor': INK2, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    rows = [(tag, kind) for tag, *_ in MODELS for kind, _ in CONDITIONS[1:]]
    y = np.arange(len(rows))[::-1].astype(float)
    labels = [f"{SHORT[tag]}, {'explicit' if kind == '_v2' else 'names'}" for tag, kind in rows]
    for ax, (measure, title) in zip(axes, [('opioid', 'Strong-opioid decision changed'), ('severity', 'Severity rating changed')]):
        style(ax)
        floor = np.array([stats['conditions'][kind]['models'][tag]['flips'][f'{measure}_floor'] for tag, kind in rows])
        ident = np.array([stats['conditions'][kind]['models'][tag]['flips'][f'{measure}_counterfactual'] for tag, kind in rows])
        ax.hlines(y, np.minimum(floor, ident), np.maximum(floor, ident), color=GRID, linewidth=2, zorder=1)
        ax.plot(floor, y, linestyle='none', marker='o', markersize=6, color=MUTED, markeredgecolor='white',
                markeredgewidth=0.8, zorder=3, label='same prompt, two seeds (noise floor)')
        ax.plot(ident, y, linestyle='none', marker='o', markersize=6, color=BLUE, markeredgecolor='white',
                markeredgewidth=0.8, zorder=4, label='reference vs other identity, same seed')
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, max(0.2, float(max(floor.max(), ident.max())) * 1.15))
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=6)
        ax.set_xlabel('share of pairs whose answer differs', fontsize=8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    stats = compute()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (HERE / 'outputs' / 'paper_stats.json').write_text(json.dumps(stats, indent=1))
    (TABLES / 'models.tex').write_text(table_models() + '\n')
    (TABLES / 'baselines.tex').write_text(table_baselines(stats) + '\n')
    (TABLES / 'differences.tex').write_text(table_differences(stats) + '\n')
    (TABLES / 'noise.tex').write_text(table_noise(stats) + '\n')
    (TABLES / 'names.tex').write_text(table_names(stats) + '\n')
    figure_differences(stats, FIGURES / 'fig2_differences.pdf')
    figure_noise(stats, FIGURES / 'fig3_noise.pdf')
    print(json.dumps(stats['totals']))
    for kind, label in CONDITIONS:
        for tag, *_ in MODELS:
            e = stats['conditions'][kind]['models'].get(tag)
            if not e:
                continue
            sev = max(e['arms'].values(), key=lambda r: abs(r['severity']['diff']))['severity'] if e['arms'] else {}
            print(f"{label:18} {SHORT[tag]:16} n={e['responses']} parsed={e['parsed']} mean={e['mean_severity']} "
                  f"opioid={e['strong_opioid_rate']} anchored={e.get('anchored_to_stated_rating', '-')} "
                  f"max|dsev|={sev.get('diff')} minHolm_sev={min(r['severity']['p_holm'] for r in e['arms'].values()):.3f} "
                  f"minHolm_op={min(r['strong_opioid']['p_holm'] for r in e['arms'].values()):.3f} "
                  f"MDE sev={e['mde_severity']['mde']} op={e['mde_strong_opioid']['mde']}")


if __name__ == '__main__':
    main()
