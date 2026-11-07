"""Figure 1: the two-study workflow, drawn to the diagram-design skill's rules.

Source of truth is the HTML this script writes (inline SVG, accessible title and description);
the SVG is extracted from it and the PDF the paper uses is converted from that SVG.

    python3.10 workflow_diagram.py                 # writes outputs/figure1/ and ../paper/figures/fig1_workflow.pdf

Style: matches the paper's charts (white page, neutral ink, the charts' blue as the single accent,
Liberation Sans). Layout: one reading order, 1 to 6, as a snake (left to right, then right to left),
so no connector crosses a label or another connector. One focal box (the paired comparison).
"""
from __future__ import annotations

import html
import re
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'outputs' / 'figure1'
PDF = HERE.parent / 'paper' / 'figures' / 'fig1_workflow.pdf'

PAPER, PANEL, INK, MUTED, SOFT, RULE = '#ffffff', '#f6f6f4', '#0b0b0b', '#52514e', '#6b6a66', '#c3c2b7'
ARROW, ACCENT, ACCENT_TINT = '#898781', '#2a78d6', '#eaf2fc'
FONT = "'Liberation Sans', 'Helvetica Neue', Arial, sans-serif"

W, H = 624, 280                       # 6.5 in at 96 dpi: 1 px = 0.75 pt, so a 9.5 px line is 7.1 pt in print
BOX_W, BOX_H = 168, 80
COLS = [24, 228, 432]
ROW_Y = [28, 152]

NODES = [  # (reading order, column, row, name, sublabel lines, focal)
    (1, 0, 0, 'Cases', ['6 authored vignettes', '50 Q-Pain cases', 'pain 0–10 · opioid · dose'], False),
    (2, 1, 0, 'Identity cues', ['4 races × 2 sexes = 8', 'stated label, or', '2 names per group'], False),
    (3, 2, 0, 'Models', ['Study 1: 4 local models', 'Study 2: 17 models,', '24 settings, local + API'], False),
    (4, 2, 1, 'Capture', ['request + response saved', 'before parsing; failures', 'kept, never dropped'], False),
    (5, 1, 1, 'Paired comparison', ['identity − White man', 'same case, seed, name', 'the case is the unit'], True),
    (6, 0, 1, 'Noise and equivalence', ['re-ask: the noise floor', 'equivalence margins', 'Holm and FDR'], False),
]


def box(order, col, row, name, lines, focal):
    x, y = COLS[col], ROW_Y[row]
    stroke = ACCENT if focal else RULE
    fill = ACCENT_TINT if focal else PANEL
    sw = 1.5 if focal else 1
    out = [f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="6" fill="{PAPER}"/>',
           f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>',
           f'<rect x="{x + 8}" y="{y + 8}" width="14" height="12" rx="2" fill="none" stroke="{ACCENT if focal else MUTED}" '
           f'stroke-width="0.8" stroke-opacity="0.6"/>',
           f'<text x="{x + 15}" y="{y + 17.2}" text-anchor="middle" font-size="8" font-weight="600" '
           f'fill="{ACCENT if focal else MUTED}">{order}</text>',
           f'<text x="{x + 30}" y="{y + 18}" font-size="11" font-weight="600" fill="{INK}">{html.escape(name)}</text>']
    for k, line in enumerate(lines):
        out.append(f'<text x="{x + 30}" y="{y + 40 + 13 * k}" font-size="9.5" fill="{MUTED}">{html.escape(line)}</text>')
    return '\n    '.join(out)


def svg() -> str:
    mid = [ROW_Y[0] + BOX_H // 2, ROW_Y[1] + BOX_H // 2]
    arrows = []
    for a, b in ((0, 1), (1, 2)):                                    # row 1, left to right
        arrows.append(f'<line x1="{COLS[a] + BOX_W}" y1="{mid[0]}" x2="{COLS[b] - 1}" y2="{mid[0]}" stroke="{ARROW}" '
                      f'stroke-width="1.2" marker-end="url(#arrow)"/>')
    cx = COLS[2] + BOX_W // 2                                         # 3 down to 4
    arrows.append(f'<line x1="{cx}" y1="{ROW_Y[0] + BOX_H}" x2="{cx}" y2="{ROW_Y[1] - 1}" stroke="{ARROW}" '
                  f'stroke-width="1.2" marker-end="url(#arrow)"/>')
    for a, b, style in ((2, 1, 'accent'), (1, 0, 'accent')):          # row 2, right to left, through the focal box
        arrows.append(f'<line x1="{COLS[a]}" y1="{mid[1]}" x2="{COLS[b] + BOX_W + 1}" y2="{mid[1]}" stroke="{ACCENT}" '
                      f'stroke-width="1.4" marker-end="url(#arrow-accent)"/>')
    boxes = '\n    '.join(box(*n) for n in NODES)
    return f'''<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="workflow-title workflow-desc" font-family="{FONT}">
  <title id="workflow-title">Two-study audit workflow</title>
  <desc id="workflow-desc">Six steps in two rows: cases, identity cues and models are varied; the responses are captured, compared in pairs against a White-man reference on the same case and seed, and judged against noise and equivalence margins.</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ARROW}"/></marker>
    <marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>
  </defs>
  <rect width="100%" height="100%" fill="{PAPER}"/>
  <text x="24" y="18" font-size="8.5" font-weight="600" letter-spacing="0.14em" fill="{SOFT}">DESIGN: WHAT WAS VARIED, WHAT WAS HELD FIXED</text>
  <text x="24" y="142" font-size="8.5" font-weight="600" letter-spacing="0.14em" fill="{SOFT}">ANALYSIS: HOW A DIFFERENCE WAS JUDGED</text>
  <g>
    {chr(10).join('    ' + a for a in arrows).strip()}
  </g>
  <g>
    {boxes}
  </g>
  <text x="24" y="268" font-size="9" fill="{SOFT}">Every count, table and figure in the paper is regenerated from the saved exchanges by script.</text>
</svg>'''


def page(svg_text: str) -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Two-study audit workflow</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: {FONT}; background: {PAPER}; color: {INK}; padding: 2rem; }}
    .frame {{ max-width: 720px; margin: 0 auto; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    svg {{ width: 100%; min-width: {W}px; display: block; }}
    @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="diagram-container">
{svg_text}
    </div>
  </div>
</body>
</html>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = svg()
    (OUT / 'fig1_workflow.html').write_text(page(s))
    (OUT / 'fig1_workflow.svg').write_text(s + '\n')
    import cairosvg
    PDF.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2pdf(bytestring=s.encode(), write_to=str(PDF))
    print(OUT / 'fig1_workflow.html', OUT / 'fig1_workflow.svg', PDF, sep='\n')


if __name__ == '__main__':
    main()
