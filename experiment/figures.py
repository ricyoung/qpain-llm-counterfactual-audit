"""Figures 2 to 5 of the paper, drawn from the saved statistics (no analysis is re-run).

One design language for every chart, following the dataviz skill's specs:
  - one hue per entity across all figures: explicit labels = blue (circle), name cues = orange
    (diamond); neutral ink for text, hairline solid grid, no dashed lines;
  - thin marks (CI 1.3 pt, markers >= 6 pt) with a surface-coloured ring so overlaps stay legible;
  - a legend whenever there are two series, labels in neutral ink, never in a series colour;
  - the equivalence margin is a shaded band, not a dashed line.

    python3.10 figures.py            # all four charts, from outputs/paper_stats.json and panel_stats.json
    python3.10 figures.py fig4       # one of fig2 fig3 fig4 fig5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FIGURES = HERE.parent / 'paper' / 'figures'

INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BAND = '#eef3fa'                                   # equivalence band: the blue ramp's lightest wash
CUE = {'explicit': ('#2a78d6', 'o', 'Explicit labels'), 'names': ('#eb6834', 'D', 'Name cues')}
MARK, CI_W, RING = 5.6, 1.3, 1.1                   # marker size (pt), interval width, surface ring width


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7.5, 'text.color': INK,
                         'axes.labelcolor': INK2, 'pdf.fonttype': 42, 'axes.linewidth': 0.8})
    return plt


def tidy(ax, xgrid=True, left=True):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.spines['left'].set_visible(left)
    ax.tick_params(colors=INK2, labelsize=7, length=2.5, width=0.8)
    if xgrid:
        ax.grid(axis='x', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def point(ax, x, y, lo, hi, cue, label=None):
    colour, marker, _ = CUE[cue]
    ax.hlines(y, lo, hi, color=colour, linewidth=CI_W, zorder=2, capstyle='round')
    ax.plot(x, y, linestyle='none', marker=marker, markersize=MARK if marker == 'o' else MARK - 0.6, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=RING, zorder=3, label=label, clip_on=False)


def cue_legend(fig, notes=(), x=0.012):
    """Legend on the first line, then one footnote per line, all placed in inches from the top edge."""
    from matplotlib.lines import Line2D
    height = fig.get_size_inches()[1]
    handles = [Line2D([], [], linestyle='none', marker=CUE[c][1], markersize=6, color=CUE[c][0],
                      markeredgecolor=SURFACE, label=CUE[c][2]) for c in ('explicit', 'names')]
    fig.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=7.5, labelcolor=INK2,
               bbox_to_anchor=(x, 1 - 0.04 / height), handletextpad=0.3, columnspacing=1.4)
    for k, text in enumerate(notes):
        fig.text(x + 0.004, 1 - (0.30 + 0.15 * k) / height, text, fontsize=6.8, color=MUTED, va='top')


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(path)


# ------------------------------------------------------------------ Figure 2 (Study 1, four local models)

def fig2(path: Path = FIGURES / 'fig2_differences.pdf') -> None:
    from make_paper_assets import ARM_LABEL, COMPARISON_ARMS, MODELS, SHORT
    stats = json.loads((HERE / 'outputs' / 'paper_stats.json').read_text())
    plt = setup()
    labels = [ARM_LABEL[a] for a in COMPARISON_ARMS]
    n = len(labels)
    y = np.arange(n)[::-1].astype(float)
    cols = [('severity', 'Pain severity (0–10 points)', 0.35, None),
            ('strong_opioid', 'Strong opioid (proportion)', 0.25, 0.10)]
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(6.5, 8.0), sharex='col',
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.1, 'left': 0.27, 'right': 0.985,
                                          'top': 0.875, 'bottom': 0.07})
    for r, (tag, *_) in enumerate(MODELS):
        for c, (field, title, limit, margin) in enumerate(cols):
            ax = axes[r, c]
            tidy(ax, left=(c == 0))
            if margin:
                ax.axvspan(-margin, margin, color=BAND, zorder=0, linewidth=0)
            ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
            for cue, dy in (('explicit', 0.2), ('names', -0.2)):
                kind = '_v2' if cue == 'explicit' else '_named'
                arms = stats['conditions'][kind]['models'][tag]['arms']
                for yy, lab in zip(y, labels):
                    a = arms[lab][field]
                    point(ax, a['diff'], yy + dy, a['ci95'][0], a['ci95'][1], cue)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-0.7, n - 0.3)
            ax.set_yticks(y)
            ax.set_yticklabels(labels if c == 0 else [], fontsize=7)
            ax.tick_params(axis='y', length=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            if r == 0:
                ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=14, fontweight='bold')
            if c == 0:
                ax.annotate(SHORT[tag], xy=(0, 1), xycoords='axes fraction', xytext=(-88, 6),
                            textcoords='offset points', ha='left', va='bottom', fontsize=8, color=INK)
    axes[-1, 0].set_xlabel('Difference from White man (points)', fontsize=7)
    axes[-1, 1].set_xlabel('Difference from White man (proportion)', fontsize=7)
    cue_legend(fig, notes=('Marker: mean paired difference; bar: bootstrap 95% CI over the six cases.',
                           'Band: ±0.10 equivalence margin (strong opioid). The ±1-point severity margin is off this scale.'))
    save(fig, path)


# ------------------------------------------------------------------ Figure 3 (noise floor)

def fig3(path: Path = FIGURES / 'fig3_noise.pdf') -> None:
    from make_paper_assets import MODELS, SHORT
    from matplotlib.lines import Line2D
    stats = json.loads((HERE / 'outputs' / 'paper_stats.json').read_text())
    plt = setup()
    rows = [(tag, kind) for tag, *_ in MODELS for kind in ('_v2', '_named')]
    pos, y = [], 0.0
    for k in range(len(rows)):
        pos.append(y)
        y -= 1.0 if k % 2 == 0 else 1.55            # a wider gap between models than between cues
    pos = np.array(pos)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.55), sharey=True,
                             gridspec_kw={'wspace': 0.12, 'left': 0.255, 'right': 0.985, 'top': 0.80, 'bottom': 0.14})
    for ax, (measure, title) in zip(axes, [('opioid', 'Strong-opioid decision changed'),
                                           ('severity', 'Severity rating changed')]):
        tidy(ax, left=False)
        floor = np.array([stats['conditions'][k]['models'][t]['flips'][f'{measure}_floor'] for t, k in rows])
        ident = np.array([stats['conditions'][k]['models'][t]['flips'][f'{measure}_counterfactual'] for t, k in rows])
        for yy, f, i, (t, k) in zip(pos, floor, ident, rows):
            cue = 'explicit' if k == '_v2' else 'names'
            ax.hlines(yy, min(f, i), max(f, i), color=GRID, linewidth=2.4, zorder=1, capstyle='round')
            ax.plot(f, yy, linestyle='none', marker='o', markersize=MARK, color=MUTED, markeredgecolor=SURFACE,
                    markeredgewidth=RING, zorder=3, clip_on=False)
            ax.plot(i, yy, linestyle='none', marker=CUE[cue][1], markersize=MARK if cue == 'explicit' else MARK - 0.6,
                    color=CUE[cue][0], markeredgecolor=SURFACE, markeredgewidth=RING, zorder=4, clip_on=False)
        ax.set_xlim(0, max(0.2, float(max(floor.max(), ident.max())) * 1.12))
        ax.set_ylim(pos.min() - 0.7, pos.max() + 0.7)
        ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=8, fontweight='bold')
        ax.set_xlabel('share of pairs whose answer differs', fontsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune='upper' if measure == 'opioid' else None))
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(pos)
    axes[0].set_yticklabels(['explicit' if k == '_v2' else 'names' for _, k in rows], fontsize=7)
    axes[0].tick_params(axis='y', length=0)
    for k in range(0, len(rows), 2):
        axes[0].annotate(SHORT[rows[k][0]], xy=(0, (pos[k] + pos[k + 1]) / 2), xycoords=('axes fraction', 'data'),
                         xytext=(-46, 0), textcoords='offset points', ha='right', va='center', fontsize=8, color=INK)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=6, color=MUTED, markeredgecolor=SURFACE,
                      label='same prompt, new seed (noise floor)'),
               Line2D([], [], linestyle='none', marker='o', markersize=6, color=CUE['explicit'][0],
                      markeredgecolor=SURFACE, label='other identity vs White man, same seed: explicit labels'),
               Line2D([], [], linestyle='none', marker='D', markersize=5.4, color=CUE['names'][0],
                      markeredgecolor=SURFACE, label='other identity vs White man, same seed: name cues')]
    fig.legend(handles=handles, loc='upper left', ncol=1, frameon=False, fontsize=7.5, labelcolor=INK2,
               bbox_to_anchor=(0.012, 0.995), handletextpad=0.3, labelspacing=0.35)
    save(fig, path)


# ------------------------------------------------------------------ Figures 4 and 5 (the 17-model panel)

CONTRAST_COLUMNS = [('race', 'Black', 'Black − White'), ('race', 'Hispanic', 'Hispanic − White'),
                    ('race', 'Asian', 'Asian − White'), ('sex', None, 'Woman − man')]
OUTCOMES = {'severity': ('Pain severity (0–10 points)', 0.05, 0.25, None),
            'offer': ('Opioid offered (proportion)', 0.05, 0.2, 0.10)}


def fig_panel(field: str, path: Path) -> bool:
    from make_panel_assets import display, order, spec_for
    stats = json.loads((HERE / 'outputs' / 'panel_stats.json').read_text())
    title, step, floor_limit, margin = OUTCOMES[field]
    labels = [l for l in order(stats) if 'qpain' in stats['configurations'][l]['sets']
              and all(c in stats['configurations'][l]['sets']['qpain'] for c in CUE)]
    if not labels:
        return False
    plt = setup()

    def result(label, cue, kind, key):
        cell = stats['configurations'][label]['sets']['qpain'][cue]
        r = cell[kind][field].get(key) if kind == 'race' else cell[kind][field]
        return r or None

    hit = {l for l in labels for cue in CUE
           for rows in stats['configurations'][l]['sets']['qpain'][cue]['identity'].get(field, {}).values()
           for r in [rows] if r.get('p_case_holm', 1.0) < 0.05}
    # group rows: local models first, then the API panel, with a header row for each group
    groups = []
    for access, name in (('local', 'Local models (Ollama)'), ('API', 'API models')):
        members = [l for l in labels if spec_for(stats['configurations'][l]['model'],
                                                 stats['configurations'][l]['think'])['access'] == access]
        if members:
            groups.append((name, members))
    ypos, y = {}, 0.0
    headers = []
    for gi, (name, members) in enumerate(groups):
        y -= 0.5 if gi == 0 else 1.1
        headers.append((name, y))
        y -= 0.9
        for l in members:
            ypos[l] = y
            y -= 1.0
    ymin, ymax = y + 0.4, 0.45
    widest = max((abs(v) for l in labels for cue in CUE for kind, key, _ in CONTRAST_COLUMNS
                  for r in [result(l, cue, kind, key)] if r for v in r['ci95']), default=0.0)
    limit = max(floor_limit, np.ceil(widest * 1.05 / step) * step)       # never clip an interval
    height = 0.205 * (-ymin) + 1.45
    fig, axes = plt.subplots(1, 4, figsize=(6.5, height), sharey=True,
                             gridspec_kw={'wspace': 0.11, 'left': 0.305, 'right': 0.985,
                                          'top': 1 - 0.86 / height, 'bottom': 0.55 / height})
    for ax, (kind, key, name) in zip(axes, CONTRAST_COLUMNS):
        tidy(ax, left=False)
        if margin:
            ax.axvspan(-margin, margin, color=BAND, zorder=0, linewidth=0)
        ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
        for _, hy in headers:
            ax.axhline(hy + 0.45, color=GRID, linewidth=0.7, zorder=0)
        for l in labels:
            for cue, dy in (('explicit', 0.17), ('names', -0.17)):
                r = result(l, cue, kind, key)
                if r:
                    point(ax, r['estimate'], ypos[l] + dy, r['ci95'][0], r['ci95'][1], cue)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_title(name, fontsize=8, color=INK, loc='left', pad=5, fontweight='bold')
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks([ypos[l] for l in labels])
    axes[0].set_yticklabels([display(stats, l) + (' †' if l in hit else '') for l in labels], fontsize=7)
    fig.canvas.draw()
    width_pt = max(t.get_window_extent().width for t in axes[0].get_yticklabels()) / fig.dpi * 72
    for name, hy in headers:
        axes[0].annotate(name, xy=(0, hy), xycoords=axes[0].get_yaxis_transform(),
                         xytext=(-width_pt - 4, 0), textcoords='offset points', ha='left', va='center',
                         fontsize=7.5, color=INK2, fontstyle='italic')
    unit = 'points' if field == 'severity' else 'proportion'
    fig.text(0.305 + (0.985 - 0.305) / 2, 0.06 / height, f'{title.split(" (")[0]}: difference from the White-man reference ({unit})',
             ha='center', va='bottom', fontsize=7, color=INK2)
    notes = ['† a within-model identity difference survived Holm correction (see text). Marker: mean of case means; bar: case-bootstrap 95% CI.',
             ('Shaded band: the ±0.10 equivalence margin; the ±1-point severity margin is far outside the severity scale.'
              if margin else 'The ±1-point equivalence margin lies far outside this axis, so no band is drawn.')]
    cue_legend(fig, notes=notes)
    save(fig, path)
    return True


def fig4(path: Path = FIGURES / 'fig4_panel_severity.pdf') -> bool:
    return fig_panel('severity', path)


def fig5(path: Path = FIGURES / 'fig5_panel_offer.pdf') -> bool:
    return fig_panel('offer', path)


if __name__ == '__main__':
    todo = sys.argv[1:] or ['fig2', 'fig3', 'fig4', 'fig5']
    for name in todo:
        globals()[name]()
