"""Inference in analyze_panel: each comparison draws from its own seed, and case-level
p-values over more than 16 cases use 200,000 random sign patterns.

    cd experiment && python3.10 -m pytest -q test_analyze_panel.py
"""
import numpy as np

import analyze_panel as ap


def cases(values) -> dict[str, list[float]]:
    return {f'case{i:02d}': [float(v)] for i, v in enumerate(values)}


def exact_sign_flip_p(values) -> float:
    v = np.asarray(values, dtype=float)
    signs = ((np.arange(2 ** len(v))[:, None] >> np.arange(len(v))) & 1) * 2 - 1
    return float(np.mean(np.abs(signs @ v / len(v)) >= abs(v.mean()) - 1e-12))


def test_a_comparison_does_not_depend_on_what_was_computed_before_it():
    per_case = cases(np.linspace(-0.2, 0.6, 30))
    first = ap.infer(per_case, 1.0)
    ap.infer(cases([i % 3 - 1.0 for i in range(25)]), 1.0)      # another comparison in between
    assert ap.infer(per_case, 1.0) == first


def test_different_data_draw_different_patterns():
    a = ap.infer(cases(np.linspace(-0.2, 0.6, 30)), 1.0)
    b = ap.infer(cases(np.linspace(-0.2, 0.6, 30) + 0.001), 1.0)
    assert a['ci95'] != b['ci95']


def test_case_level_p_uses_200000_sign_patterns():
    # Only the all-positive pattern (and its mirror) reaches the observed mean of 40 equal
    # values, so the Monte Carlo p is 1 / (draws + 1).
    assert ap.B_FLIP == 200_000
    assert ap.infer(cases([1.0] * 40), 1.0)['p_case'] == round(1 / (ap.B_FLIP + 1), 6)


def test_monte_carlo_p_agrees_with_the_exact_p():
    values = [0.9, -0.4, 0.7, 0.3, -0.2, 0.8, 0.5, -0.6, 0.4, 0.1, 0.6, -0.3, 0.2, 0.9, -0.1, 0.5, 0.3, -0.5]
    exact = exact_sign_flip_p(values)                           # 2^18 patterns
    assert 0.005 < exact < 0.2
    assert abs(ap.infer(cases(values), 1.0)['p_case'] - exact) < 0.003   # about 5 Monte Carlo SEs


def test_small_case_sets_stay_exact():
    values = [0.3, 0.1, -0.2, 0.4, 0.5, 0.2]
    assert ap.infer(cases(values), 1.0)['p_case'] == round(exact_sign_flip_p(values), 6)


def test_infer_stores_the_standard_error_and_the_equivalence_bound():
    values = [0.9, -0.4, 0.7, 0.3, -0.2, 0.8, 0.5, -0.6, 0.4, 0.1, 0.6, -0.3, 0.2, 0.9, -0.1, 0.5, 0.3, -0.5]
    r = ap.infer(cases(values), 1.0)
    assert r['se'] == round(float(np.std(values, ddof=1) / np.sqrt(len(values))), 6)
    # an interval fits inside a margin m exactly when its larger absolute end is below m
    assert r['bound90'] == round(max(abs(r['ci90'][0]), abs(r['ci90'][1])), 6) or abs(r['bound90'] - max(abs(r['ci90'][0]), abs(r['ci90'][1]))) < 1e-4
    assert r['equivalent'] == (r['bound90'] < 1.0)
