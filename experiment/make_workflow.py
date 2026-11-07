"""Draw the methods workflow (Figure 1).

Layout: a design row (stages 1-4, left to right), one drop, and an analysis row
(stages 5-8, right to left), so no connector crosses a box. Header bands are
clipped to each box's rounded outline, and every text line is measured against
its box after drawing: the script refuses to save a figure whose text overflows.

    python3.10 make_workflow.py              ->  ../paper/figures/fig1_workflow.pdf
    python3.10 make_workflow.py --study1     (the Study 1 version: authored cases, local models)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent.parent / 'paper' / 'figures' / 'fig1_workflow.pdf'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
BLUE, SLATE, FILL, EDGE = '#2a78d6', '#52514e', '#f6f6f3', '#c3c2b7'


def stages(study1_only: bool) -> list[tuple[str, list[str]]]:
    if study1_only:
        return [
            ('Vignettes', ['6 acute-pain cases: renal colic,', '   femur fracture, sickle cell, back',
                           '   pain, migraine, post-operative', 'LLM-drafted, author-edited',
                           'observed behaviour and vital signs;', 'no self-reported pain rating']),
            ('Identity cues', ['4 races × 2 sexes: 8 identities;', '   reference: White man',
                               'explicit: "a 38-year-old Black', '   woman"',
                               'name only: "Lakisha Washington"', '   (2 names per identity)']),
            ('Local inference', ['4 open-weight models, 3B to 15B,', '   4-bit, served by Ollama',
                                 'temperature 0.7, seeds 0 to 4', 'JSON answer: pain severity 0–10',
                                 '   and analgesic class']),
            ('Responses and scoring', ['responses saved before parsing,', '   keyed by model, case, identity,',
                                       '   name and seed', 'severity 0–10; strong-opioid',
                                       '   indicator; invalid responses', '   counted']),
            ('Paired comparison', ['identity minus reference on the', '   same case, seed and name slot',
                                   'bootstrap 95% CI', 'two-sided sign-flip permutation p',
                                   'Holm correction over 7 identities']),
            ('Robustness', ['noise floor: same prompt, 2 seeds', 'matched cross-seed identity flips',
                            'case-level (cluster) bootstrap', 'equivalence within ±1 point and',
                            '   ±0.10 strong-opioid', 'per-name means']),
        ]
    return [
        ('Cases', ['6 authored acute-pain cases', '   (LLM-drafted, author-edited)',
                   '50 Q-Pain cases written by a', '   clinical expert, 5 pain contexts',
                   'outcome: severity 0–10, plus', '   opioid class, or offer and dose']),
        ('Identity cues', ['4 races × 2 sexes: 8 identities;', '   reference: White man',
                           'explicit: "a 38-year-old Black', '   woman"',
                           'name only: 2 names per identity', '   (authored), 10 (Q-Pain)']),
        ('Models', ['Study 1: 4 open-weight models,', '   3B to 15B, 4-bit, local (Ollama)',
                    'Study 2: local and API models,', '   provider and precision pinned',
                    'reasoning on vs off; 4-bit vs bf16', 'temperature 0.7, fixed seeds']),
        ('Capture and scoring', ['request and response saved', '   in full before parsing; failures',
                                 '   logged and re-requested',
                                 'JSON fields scored; invalid',
                                 '   responses counted, not dropped']),
        ('Paired comparison', ['identity minus reference on the', '   same case, seed and name slot',
                               'case-level bootstrap 95% CI', 'sign-flip p; Holm over 7 identities;',
                               '   false discovery rate over all']),
        ('Robustness', ['noise floor: same prompt, new seed', 'equivalence within ±1 point, ±0.10',
                        'race and sex pooled contrasts', 'reasoning on vs off; 4-bit vs bf16',
                        'per-name means']),
    ]


def draw_stage(ax, x, y, w, h, number, title, lines, accent, texts):
    body = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.14',
                          linewidth=0.8, edgecolor=EDGE, facecolor=FILL, zorder=1)
    ax.add_patch(body)
    band = Rectangle((x, y + h - 0.46), w, 0.46, facecolor=accent, edgecolor='none', zorder=2)
    ax.add_patch(band)
    band.set_clip_path(body)                       # the band follows the rounded corners exactly
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.14',
                                linewidth=0.8, edgecolor=EDGE, facecolor='none', zorder=3))
    texts.append((ax.text(x + 0.16, y + h - 0.23, f'{number}  {title}', ha='left', va='center',
                          fontsize=7.6, color='white', weight='bold', zorder=4), (x, y, w, h)))
    texts.append((ax.text(x + 0.16, y + h - 0.66, '\n'.join(lines), ha='left', va='top', fontsize=6.5,
                          color=INK, linespacing=1.4, zorder=4), (x, y, w, h)))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10, linewidth=1.1,
                                 color=INK2, shrinkA=0, shrinkB=0, zorder=5))


def check_fit(fig, ax, texts) -> list[str]:
    """Every text block must sit inside its box with a small margin."""
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    problems = []
    for text, (x, y, w, h) in texts:
        (x0, y0), (x1, y1) = inverse.transform(text.get_window_extent(renderer).get_points())
        if x0 < x + 0.05 or x1 > x + w - 0.05 or y0 < y + 0.05 or y1 > y + h:
            problems.append(f'{text.get_text().splitlines()[0]!r} spans x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}; '
                            f'box x {x:.2f}-{x + w:.2f}, y {y:.2f}-{y + h:.2f}')
    return problems


def main(study1_only: bool) -> int:
    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.0); ax.axis('off')
    w, h, gap = 3.9, 2.75, 0.6
    xs = [0.05 + i * (w + gap) for i in range(3)]
    top, bottom = 4.65, 0.8
    texts = []
    spec = stages(study1_only)
    for i, (title, lines) in enumerate(spec[:3]):                 # design row, left to right
        draw_stage(ax, xs[i], top, w, h, i + 1, title, lines, BLUE, texts)
    for i, (title, lines) in enumerate(spec[3:]):                 # analysis row, right to left
        draw_stage(ax, xs[2 - i], bottom, w, h, i + 4, title, lines, SLATE, texts)
    for i in range(2):
        arrow(ax, (xs[i] + w + 0.05, top + h / 2), (xs[i + 1] - 0.05, top + h / 2))
        arrow(ax, (xs[2 - i] - 0.05, bottom + h / 2), (xs[1 - i] + w + 0.05, bottom + h / 2))
    arrow(ax, (xs[2] + w / 2, top - 0.05), (xs[2] + w / 2, bottom + h + 0.05))   # the single drop
    ax.text(0.05, top + h + 0.25, 'Design: what was varied and what was held fixed', ha='left', va='center',
            fontsize=7, color=MUTED, style='italic')
    ax.text(0.05, bottom + h + 0.25, 'Analysis: how a difference was judged', ha='left', va='center',
            fontsize=7, color=MUTED, style='italic')
    ax.text(6.5, 0.3, 'Every table, figure and number is regenerated from the saved exchanges by script; code and data are released.',
            ha='center', va='center', fontsize=6.6, color=INK2)
    fig.canvas.draw()
    problems = check_fit(fig, ax, texts)
    if problems:
        print('text overflows its box; figure not saved:\n  ' + '\n  '.join(problems), file=sys.stderr)
        return 1
    fig.savefig(OUT, bbox_inches='tight', pad_inches=0.03)
    print(OUT)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study1', action='store_true', help='the Study 1 version of the figure')
    sys.exit(main(parser.parse_args().study1))
