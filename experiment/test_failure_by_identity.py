"""Do invalid answers depend on the identity in the prompt?  (review item S11)

flags is an (n_blocks, 8) 0/1 array: one row per case-and-seed block, one column per identity,
1 where that identity's response was invalid. The test permutes the flags within each row, so a
case that makes a model fail for every identity cannot look like an identity effect.

    cd experiment && python3.10 -m pytest -q test_failure_by_identity.py
"""
import numpy as np

import failure_by_identity as fb


def test_failures_concentrated_in_one_identity_are_detected():
    flags = np.zeros((40, 8), dtype=int)
    flags[:, 0] = 1                                 # the first identity fails in every block
    r = fb.identity_failure_test(flags)
    assert r['invalid'] == 40 and r['by_arm'][0] == 40
    assert r['p'] < 0.001


def test_failures_spread_evenly_across_identities_are_not():
    flags = np.zeros((40, 8), dtype=int)
    for i in range(40):
        flags[i, i % 8] = 1                         # one failure per block, rotating through the identities
    r = fb.identity_failure_test(flags)
    assert r['chi2'] == 0.0 and r['p'] == 1.0


def test_a_case_that_fails_for_every_identity_is_not_an_identity_effect():
    flags = np.zeros((40, 8), dtype=int)
    flags[:10, :] = 1                               # ten hard blocks fail for all eight identities
    r = fb.identity_failure_test(flags)
    assert r['chi2'] == 0.0 and r['p'] == 1.0


def test_no_failures_gives_p_one_and_the_result_is_reproducible():
    assert fb.identity_failure_test(np.zeros((12, 8), dtype=int))['p'] == 1.0
    flags = (np.arange(96).reshape(12, 8) % 5 == 0).astype(int)
    assert fb.identity_failure_test(flags) == fb.identity_failure_test(flags)
