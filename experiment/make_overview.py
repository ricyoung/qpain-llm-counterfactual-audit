"""Draw the study-overview schematic (Figure 1): what was varied, what was held fixed,
what was measured and how it was analysed. Pure matplotlib, no data dependency.

    python3.10 make_overview.py   ->  ../paper/figures/fig0_overview.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / 'paper' / 'figures' / 'fig1_overview.pdf'
INK, INK2, MUTED, BLUE, ORANGE, FILL, EDGE = '#0b0b0b', '#52514e', '#898781', '#2a78d6', '#eb6834', '#f4f4f1', '#c3c2b7'


def box(ax, x, y, w, h, title, lines, accent=BLUE, title_size=7.6, body_size=6.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                linewidth=0.8, edgecolor=EDGE, facecolor=FILL))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.40), w, 0.40, boxstyle='round,pad=0,rounding_size=0.12',
                                linewidth=0, facecolor=accent))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.40), w, 0.2, boxstyle='square,pad=0', linewidth=0, facecolor=accent))  # square the lower corners
    ax.text(x + w / 2, y + h - 0.20, title, ha='center', va='center', fontsize=title_size, color='white', weight='bold')
    ax.text(x + 0.12, y + h - 0.62, '\n'.join(lines), ha='left', va='top', fontsize=body_size, color=INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, linewidth=1.0, color=INK2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.16, label, ha='center', va='bottom', fontsize=7, color=INK2)


def main() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.set_xlim(-0.1, 13.1); ax.set_ylim(0, 8.0); ax.axis('off')

    # Row 1: the factorial design. Three factors, multiplied.
    y1, h1 = 4.7, 3.15
    box(ax, 0.15, y1, 3.0, h1, 'Six pain vignettes', [
        'renal colic (38)', 'femur fracture (45)', 'sickle cell crisis (24)',
        'low back pain (41)', 'migraine (33)', 'post-appendectomy (29)',
        '', 'history, observed behaviour,', 'vitals, no allergies; no self-rating'])
    ax.text(3.45, y1 + h1 / 2, '×', ha='center', va='center', fontsize=14, color=INK2)
    box(ax, 3.75, y1, 4.1, h1, 'Eight identities, two cue types', [
        '4 races × 2 sexes; reference = White man',
        '',
        'Explicit label  (960 responses)',
        '   "Patient: a 38-year-old Black woman."',
        'Name cue  (1,920 responses)',
        '   "Patient: Lakisha Washington, age 38."',
        '   two names per identity',
        '', 'every other token of the prompt fixed'])
    ax.text(8.15, y1 + h1 / 2, '×', ha='center', va='center', fontsize=14, color=INK2)
    box(ax, 8.45, y1, 4.4, h1, 'Four open-weight models', [
        'gemma3 12B (Google)', 'qwen3 14B (Alibaba, thinking off)',
        'ministral-3 14B (Mistral AI)', 'llama3.2 3B (Meta)',
        '', 'Ollama, Q4_K_M, local GPUs', 'temperature 0.7, seeds 0–4',
        'JSON-only answer, raw response saved'])

    # Row 2: outcome -> paired analysis -> context.
    y2, h2 = 0.35, 3.4
    box(ax, 0.15, y2, 3.6, h2, 'Outcomes per response', [
        'pain_severity: integer 0–10',
        'analgesic: none | non-opioid |',
        '   weak-opioid | strong-opioid',
        '   → strong-opioid indicator',
        '', '3,840 responses, 100% parsed:', 'pilot 960, explicit 960, names 1,920'], accent=INK2)
    box(ax, 4.35, y2, 4.1, h2, 'Paired comparison', [
        'identity − White-man reference',
        'on the same case, seed (and name slot)',
        '30 pairs/identity (explicit), 60 (names)',
        '', 'mean difference, bootstrap 95% CI',
        'two-sided sign-flip permutation p',
        'Holm over the 7 identities per model'], accent=INK2)
    box(ax, 9.05, y2, 3.8, h2, 'Context for the estimate', [
        'noise floor: same prompt, two seeds,',
        '   against identity change, same seed',
        'minimum detectable difference',
        '   (80% power, two-sided α = 0.05)',
        'per-name means (16 names) and',
        '   differences by reference name'], accent=INK2)

    # Arrows: the design feeds the outcomes; outcomes feed the analysis; analysis feeds context.
    arrow(ax, 6.5, y1 - 0.05, 2.6, y2 + h2 + 0.05)
    arrow(ax, 3.8, y2 + h2 / 2, 4.3, y2 + h2 / 2)
    arrow(ax, 8.5, y2 + h2 / 2, 9.0, y2 + h2 / 2)
    ax.text(6.5, y1 - 0.18, 'one request per vignette × identity × model × seed', ha='center', va='top', fontsize=6.8, color=INK2)

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
