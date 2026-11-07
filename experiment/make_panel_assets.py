"""Tables and figures for the model-panel study, built from outputs/panel_stats.json
(run analyze_panel.py first). Nothing here is typed by hand.

  ../paper/tables/panel_models.tex       every configuration: access, precision, counts, cost
  ../paper/tables/panel_<set>.tex        per configuration and cue: baselines, largest
                                         identity difference, Holm p, equivalence count
  ../paper/tables/panel_thinking.tex     reasoning on vs off (or high vs low effort)
  ../paper/tables/panel_precision.tex    gemma 3 12B at 4-bit (local) vs bf16 (API)
  ../paper/figures/fig4_panel_<cue>.pdf  Q-Pain race and sex contrasts, every configuration

    python3.10 make_panel_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
STATS = HERE / 'outputs' / 'panel_stats.json'
PANEL = json.loads((HERE / 'panel.json').read_text())
TABLES, FIGURES = HERE.parent / 'paper' / 'tables', HERE.parent / 'paper' / 'figures'

SHORT = {
    'gemma3:12b': 'gemma3 12B', 'qwen3:14b': 'qwen3 14B', 'ministral-3:14b': 'ministral-3 14B',
    'llama3.2:latest': 'llama3.2 3B', 'qwen3.8:27b': 'qwen3.8 27B', 'gpt-oss:20b': 'gpt-oss 20B',
    'qwen3.6:35b': 'qwen3.6 35B', 'openai/gpt-4.1-mini': 'GPT-4.1 mini', 'anthropic/claude-haiku-4.5': 'Claude Haiku 4.5',
    'google/gemini-2.5-flash': 'Gemini 2.5 Flash', 'z-ai/glm-4.6': 'GLM-4.6',
    'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B', 'deepseek/deepseek-v4.1-flash': 'DeepSeek V4.1 Flash',
    'qwen/qwen3-235b-a22b-2507': 'Qwen3 235B-A22B', 'openai/gpt-oss-120b': 'gpt-oss 120B',
    'google/gemma-3-12b-it': 'Gemma 3 12B', 'beam:cloud': 'Beam 501B-A23B'}
MODE = {'on': 'reasoning on', 'off': 'reasoning off', 'low': 'effort low', 'medium': 'effort medium', 'high': 'effort high'}
SET_LABEL = {'authored': 'Authored cases', 'qpain': 'Q-Pain'}
FIELD_LABEL = {'severity': 'severity', 'strong_opioid': 'strong opioid', 'offer': 'opioid offered', 'high_dose': 'high dose'}
CONTRASTS = [('race', 'Black', 'Black − White'), ('race', 'Hispanic', 'Hispanic − White'),
             ('race', 'Asian', 'Asian − White'), ('sex', None, 'woman − man')]
# dataviz reference palette, categorical slots 1-4 (validated light mode; aqua and yellow
# fall below 3:1 on white, so each also has its own marker and the table carries the values)
COLOURS, MARKERS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], ['o', 'D', 's', '^']
GRID, AXIS, INK, INK2, MUTED = '#e1e0d9', '#c3c2b7', '#0b0b0b', '#52514e', '#898781'


def order(stats: dict) -> list[str]:
    """Configurations in panel order: local models first, then the API panel."""
    wanted = [(e['name'], t) for e in PANEL['local'] for t in e['think']]
    wanted += [(e['id'], {None: 'off'}.get(e.get('reasoning'), e.get('reasoning'))) for e in PANEL['openrouter']]
    wanted += [(e['name'], t) for e in PANEL.get('ollama_cloud', []) for t in e['think']]
    labels = [f'{m} [{t}]' for m, t in wanted]
    present = [l for l in labels if l in stats['configurations']]
    return present + sorted(set(stats['configurations']) - set(present))


def display(stats: dict, label: str) -> str:
    entry = stats['configurations'][label]
    model, think = entry['model'], entry['think']
    modes = {e['think'] for e in stats['configurations'].values() if e['model'] == model}
    name = SHORT.get(model, model)
    if model == 'google/gemma-3-12b-it':
        name += ' (bf16)'
    return name + (f', {MODE[think]}' if len(modes) > 1 or think not in ('off',) else '')


def spec_for(model: str, think: str) -> dict:
    for e in PANEL['local']:
        if e['name'] == model:
            return {'access': 'local', 'provider': 'Ollama', 'precision': e['precision']}
    for e in PANEL['openrouter']:
        if e['id'] == model:
            return {'access': 'API', 'provider': e['provider'].split('/')[0],
                    'precision': e.get('quantization') or ('fp4' if 'fp4' in e['provider'] else 'not disclosed')}
    for e in PANEL.get('ollama_cloud', []):
        if e['name'] == model:
            return {'access': 'API', 'provider': e['provider'], 'precision': e['precision']}
    return {'access': '?', 'provider': '?', 'precision': '?'}


def tex(text: str) -> str:
    return text.replace('_', r'\_').replace('−', '$-$').replace('&', r'\&')


def fmt_p(p: float) -> str:
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def signed(value: float, digits: int) -> str:
    return f'{value:+.{digits}f}'.replace('-', '$-$')


# ---------------------------------------------------------------- tables

def table_models(stats: dict) -> str:
    lines = [r'\begin{tabular}{lllllrrrr}', r'\toprule',
             r'Configuration & Access & Provider & Precision & Reasoning & Responses & Invalid & Output tokens & Cost (USD) \\',
             r'\midrule']
    for label in order(stats):
        e = stats['configurations'][label]
        s = spec_for(e['model'], e['think'])
        records = sum(c['records'] for cues in e['sets'].values() for c in cues.values())
        invalid = sum(c['invalid'] for cues in e['sets'].values() for c in cues.values())
        cost = e['usage'].get('cost_usd', 0.0)
        tokens = e['usage'].get('mean_completion_tokens')
        tokens = '--' if tokens is None else f'{tokens:.0f}'
        lines.append(f"{tex(display(stats, label))} & {s['access']} & {tex(s['provider'])} & {tex(s['precision'])} & "
                     f"{e['think']} & {records:,} & {invalid} & {tokens} & "
                     f"{'--' if s['access'] == 'local' else f'{cost:.2f}'} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def table_set(stats: dict, vset: str) -> str:
    fields = ('severity', 'strong_opioid') if vset == 'authored' else ('severity', 'offer', 'high_dose')
    head = ' & '.join(f'{FIELD_LABEL[f].capitalize()}' for f in fields)
    lines = [r'\begin{tabular}{ll' + 'r' * len(fields) + 'lrlrc}', r'\toprule',
             r' & & \multicolumn{' + str(len(fields)) + r'}{c}{Reference (White man) mean} & '
             r'\multicolumn{2}{c}{Largest severity $\Delta$} & \multicolumn{2}{c}{Largest '
             + FIELD_LABEL[fields[1]] + r' $\Delta$} & Equivalent \\',
             r'\cmidrule(lr){3-' + str(2 + len(fields)) + r'}\cmidrule(lr){' + str(3 + len(fields)) + '-'
             + str(4 + len(fields)) + r'}\cmidrule(lr){' + str(5 + len(fields)) + '-' + str(6 + len(fields)) + '}',
             r'Configuration & Cue & ' + head + r' & $\Delta$ (identity) & $p_{\mathrm{Holm}}$ & $\Delta$ (identity) & '
             r'$p_{\mathrm{Holm}}$ & of 7 \\', r'\midrule']
    for label in order(stats):
        cues = stats['configurations'][label]['sets'].get(vset, {})
        for i, cue in enumerate(c for c in ('explicit', 'names') if c in cues):
            cell = cues[cue]
            base = ' & '.join(f"{cell['baseline'][f]:.2f}" for f in fields)
            parts = []
            for field, digits in ((fields[0], 2), (fields[1], 3)):
                rows = cell['identity'].get(field, {})
                if not rows or all(r['estimate'] == 0 for r in rows.values()):
                    parts += ['0 (all)', '1.000']
                    continue
                arm, row = max(rows.items(), key=lambda kv: abs(kv[1]['estimate']))
                parts += [f"{signed(row['estimate'], digits)} ({arm})", fmt_p(min(r['p_case_holm'] for r in rows.values()))]
            equivalent = '/'.join(str(sum(r['equivalent'] for r in cell['identity'].get(f, {}).values())) for f in fields)
            lines.append(f"{tex(display(stats, label)) if i == 0 else ''} & {cue} & {base} & " + ' & '.join(parts)
                         + f' & {equivalent} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def _ci(row: dict, digits: int) -> str:
    return f"{signed(row['estimate'], digits)} [{signed(row['ci95'][0], digits)}, {signed(row['ci95'][1], digits)}]"


def table_contrasts(stats: dict, section: str) -> str:
    """Thinking or precision contrasts: baseline shifts and identity-difference changes."""
    lines = [r'\begin{tabular}{lllllr}', r'\toprule',
             r'Contrast & Cases & Cue & $\Delta$ baseline severity [95\% CI] & $\Delta$ baseline opioid [95\% CI] & '
             r'min $p_{\mathrm{Holm}}$ \\', r'\midrule']
    items = stats[section].items() if section == 'thinking' else [('Gemma 3 12B: bf16 minus 4-bit', stats['precision'])]
    for name, result in items:
        if section == 'thinking':
            model, rest = name.split(': ')
            name = f"{SHORT.get(model, model)}: {rest}"
        first = True
        for vset in ('authored', 'qpain'):
            for cue, c in result.get(vset, {}).items():
                opioid = 'strong_opioid' if vset == 'authored' else 'offer'
                shift = c['baseline_shift']
                if not shift.get('severity') and not shift.get(opioid):
                    continue                     # fewer shared cases than analyze_panel.MIN_CASES
                p = min((r['p_case_holm'] for rows in c['identity'].values() for r in rows.values()), default=1.0)
                lines.append(f"{tex(name) if first else ''} & {SET_LABEL[vset]} & {cue} & "
                             f"{_ci(shift['severity'], 2) if shift.get('severity') else '--'} & "
                             f"{_ci(shift[opioid], 3) if shift.get(opioid) else '--'} & {fmt_p(p)} \\\\")
                first = False
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# ---------------------------------------------------------------- figure

def figure_panel(stats: dict, cue: str, path: Path) -> bool:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'text.color': INK,
                         'axes.labelcolor': INK2, 'pdf.fonttype': 42})
    labels = [l for l in order(stats) if cue in stats['configurations'][l]['sets'].get('qpain', {})]
    if not labels:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 0.3 * len(labels) + 1.1), sharey=True)
    y = np.arange(len(labels))[::-1].astype(float)
    offsets = np.linspace(0.27, -0.27, len(CONTRASTS))
    for ax, (field, title, limit) in zip(axes, [('severity', 'Pain severity (0–10 points)', 0.5),
                                                 ('offer', 'Opioid offered (proportion)', 0.20)]):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=7.5, length=3, width=0.8)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        widest = max((abs(v) for l in labels
                      for k, (kind, key, _) in enumerate(CONTRASTS)
                      for r in [stats['configurations'][l]['sets']['qpain'][cue][kind][field].get(key)
                                if kind == 'race' else stats['configurations'][l]['sets']['qpain'][cue][kind][field]]
                      if r for v in r['ci95']), default=0.0)
        limit = max(limit, np.ceil(widest * 1.05 / 0.05) * 0.05)   # never clip an interval
        for row, label in zip(y, labels):
            cell = stats['configurations'][label]['sets']['qpain'][cue]
            for k, (kind, key, name) in enumerate(CONTRASTS):
                r = cell[kind][field][key] if kind == 'race' else cell[kind][field]
                if not r:
                    continue
                lo, hi = np.clip(r['ci95'], -limit, limit)
                ax.hlines(row + offsets[k], lo, hi, color=COLOURS[k], linewidth=1.2, zorder=2)
                ax.plot(np.clip(r['estimate'], -limit, limit), row + offsets[k], linestyle='none', marker=MARKERS[k],
                        markersize=4.2, color=COLOURS[k], markeredgecolor='white', markeredgewidth=0.6, zorder=3,
                        label=name if row == y[0] else None)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([display(stats, l) for l in labels])
        ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=6)
    fig.supxlabel('Paired difference on the same Q-Pain case and seed (mean and case-bootstrap 95% CI)',
                  fontsize=8, color=INK2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=4, frameon=False, fontsize=7.5, markerscale=1.6,
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (0.3 * len(labels) + 1.1)))
    fig.savefig(path)
    plt.close(fig)
    return True


def numbers(stats: dict) -> str:
    """LaTeX macros for every count the text quotes, so no number is typed by hand."""
    configs = stats['configurations']
    cells = [c for e in configs.values() for cues in e['sets'].values() for c in cues.values()]
    ledger = HERE / 'outputs' / 'openrouter_ledger.jsonl'
    spent = sum(json.loads(l).get('cost', 0.0) for l in ledger.read_text().splitlines() if l.strip()) if ledger.exists() else 0.0
    api = [e for e in configs.values() if spec_for(e['model'], e['think'])['access'] == 'API']
    sw = stats['study_wide']
    values = {
        'nConfigs': len(configs), 'nConfigsLocal': len(configs) - len(api), 'nConfigsAPI': len(api),
        'nConfigsQpain': sum('qpain' in e['sets'] for e in configs.values()),
        'nModels': len({e['model'] for e in configs.values()}),
        'nResponses': f"{sum(c['records'] for c in cells):,}", 'nInvalid': f"{sum(c['invalid'] for c in cells):,}",
        'nResponsesAPI': f"{sum(c['records'] for e in api for cues in e['sets'].values() for c in cues.values()):,}",
        'costAPI': f"{sum(e['usage'].get('cost_usd', 0.0) for e in api):.2f}", 'costLedger': f'{spent:.2f}',
        'nTests': f"{sw['tests']:,}", 'nTestsBelow': sw['p_below_05'], 'nTestsExpected': sw['expected_under_null'],
        'nBHhits': len(sw['bh_q_below_05'])}
    values.update(finding_numbers(stats))
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in values.items())


def _range(stats: dict, values: dict, digits: int) -> str:
    """'low (model) to high (model)' over {configuration label: value}."""
    lo, hi = min(values, key=values.get), max(values, key=values.get)
    return (f"{values[lo]:.{digits}f} ({tex(display(stats, lo))}) to "
            f"{values[hi]:.{digits}f} ({tex(display(stats, hi))})")


def finding_numbers(stats: dict) -> dict:
    """Counts and ranges the Results quote, recomputed from the statistics on every build."""
    configs = stats['configurations']
    qp = {label: e['sets']['qpain'] for label, e in configs.items() if 'qpain' in e['sets']}
    rows = [(label, vset, cue, field, r) for label, e in configs.items() for vset, cues in e['sets'].items()
            for cue, cell in cues.items() for field, arms in cell['identity'].items() for r in arms.values()]
    qrows = [x for x in rows if x[1] == 'qpain']
    out = {'nHolmHits': sum(r.get('p_case_holm', 1.0) < 0.05 for *_, r in rows),
           'nSevNonEquiv': sum(not r['equivalent'] for _, _, _, f, r in rows if f == 'severity'),
           'nSevDiffs': f"{sum(f == 'severity' for _, _, _, f, _ in rows):,}",
           'nQpainDiffs': sum(f == 'offer' for _, _, _, f, _ in qrows),
           'nOfferNonEquiv': sum(not r['equivalent'] for _, _, _, f, r in qrows if f == 'offer'),
           'nHighNonEquiv': sum(not r['equivalent'] for _, _, _, f, r in qrows if f == 'high_dose'),
           'maxSevQpain': f"{max(abs(r['estimate']) for _, _, _, f, r in qrows if f == 'severity'):.2f}",
           'maxOfferQpain': f"{max(abs(r['estimate']) for _, _, _, f, r in qrows if f == 'offer'):.3f}",
           'maxHighQpain': f"{max(abs(r['estimate']) for _, _, _, f, r in qrows if f == 'high_dose'):.3f}"}
    for cue, name in (('explicit', 'Explicit'), ('names', 'Names')):
        black = [cues[cue]['race']['severity']['Black']['estimate'] for cues in qp.values()
                 if cue in cues and cues[cue]['race']['severity'].get('Black')]
        out[f'nBlackSevPos{name}'] = sum(v > 0 for v in black)
        out[f'nPooled{name}'] = len(black)
        out[f'medBlackSev{name}'] = f"{np.median(black):+.2f}" if black else '--'
        out[f'maxBlackSev{name}'] = f"{max(black):+.2f}" if black else '--'
    explicit = {label: cues['explicit'] for label, cues in qp.items() if 'explicit' in cues}
    out['rangeSevBase'] = _range(stats, {l: c['baseline']['severity'] for l, c in explicit.items()}, 2)
    out['rangeOfferBase'] = _range(stats, {l: c['baseline']['offer'] for l, c in explicit.items()}, 2)
    out['rangeHighBase'] = _range(stats, {l: c['baseline']['high_dose'] for l, c in explicit.items()}, 2)
    out['rangeEtaQpain'] = _range(stats, {l: min(c['eta_squared_case'] for c in cues.values())
                                          for l, cues in qp.items()}, 2)
    out['rangeFullMatch'] = _range(stats, {l: np.mean([c['reference_agreement']['full_match'] for c in cues.values()])
                                           for l, cues in qp.items()}, 2)
    return out


def main() -> None:
    stats = json.loads(STATS.read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = {'panel_models.tex': table_models(stats), 'panel_authored.tex': table_set(stats, 'authored'),
               'panel_qpain.tex': table_set(stats, 'qpain'), 'panel_thinking.tex': table_contrasts(stats, 'thinking'),
               'panel_precision.tex': table_contrasts(stats, 'precision'), 'panel_numbers.tex': numbers(stats)}
    for name, body in written.items():
        (TABLES / name).write_text(body + '\n')
        print(TABLES / name)
    for cue in ('explicit', 'names'):
        path = FIGURES / f'fig4_panel_{cue}.pdf'
        if figure_panel(stats, cue, path):
            print(path)


if __name__ == '__main__':
    main()
