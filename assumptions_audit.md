# Assumptions audit, 2026-11-05

Asked for by Ric after reading v2. Each item: the assumption, why it matters, what was
done, and the measured result (`experiment/audit_analyses.py` -> `outputs/audit_stats.json`).

## Errors found and corrected in the manuscript

1. **"H1 and H2 were formalised before the name-cue results were analysed."** False.
   The project log recorded the name-cue design before the run, but H1/H2 were written
   in the manuscript rewrite after all three conditions' results were known.
   Corrected: hypotheses stated post hoc, analyses exploratory.
2. **"The authors wrote six cases."** Incomplete. They were drafted with an LLM (Claude)
   and edited by Ric, with no clinician review. Now disclosed in Methods and Limitations.
3. **"Written after the training cut-offs, so no vignette can appear in training data."**
   Overstated: the exact text can't, but the presentations are textbook cases. Softened.
4. **"Identity changed the opioid decision no more often than re-asking."** Flattered by
   construction: identity pairs shared a seed, floor pairs did not. With the matched
   (different-seed) comparison the identity flip rate is *about equal to* the floor,
   slightly above it for gemma3 under name cues (opioid 0.118 vs 0.083; severity 0.130
   vs 0.071). Wording corrected throughout.
5. **"No difference larger than 4-14 points of opioid use."** Overstated: it treated
   30-60 seed/name replicates as independent. With the case as the unit (6 clusters)
   the opioid 95% intervals reach -0.25 and +0.20. Corrected.

## Assumptions tested with existing data

| Assumption | Result |
|---|---|
| Seeds are independent observations | No: 6 vignettes are the effective sample. Cluster bootstrap added. |
| "Not significant" means "no difference" | Severity: every comparison **equivalent to zero within +/-1 point** (TOST, 90% cluster CI; margin = half Farrar 2001's ~2-point MCID). Opioid: **not** shown equivalent within +/-0.10 for any model. |
| Models actually assess the case | Case explains 86-100% of severity variance; only 2-4 distinct severity values used. llama3.2 gives one fixed rating per case (eta2 = 1.0): its severity null is uninformative. |
| Sickle cell (race-associated condition) doesn't drive results | Excluding it changes nothing (largest severity mean +0.140). |
| Multiplicity handled | 112 identity comparisons; 1 uncorrected p < .05, fewer than the 5.6 expected by chance. |

## Limitations acknowledged, not testable without new data

- One age per case: age bias untested.
- One prompt wording, one physician role, temperature 0.7 only (instrument effects, Vohra & Ravikiran 2026).
- 4-bit quantised weights; full-precision behaviour may differ.
- `format: json` constrained decoding may suppress reasoning; free-text untested.
- Reference group fixed at White man; minority-vs-minority contrasts not reported.
- Hispanic and Asian names unvalidated; "Asian" names mix Japanese and Chinese origin.
- Comparison with Young et al. 2025 is from the abstract; full text not yet read.
- Deployment claim ("hospitals run open-weight models") rests on one radiology report.

## Being addressed by runs now in progress

- Author-written, few cases -> **Q-Pain**: 50 clinician-written vignettes, 5 pain contexts,
  Q-Pain's own names (10 per group), schema-constrained JSON (9,600 calls).
- Small models only, thinking disabled -> **scale stage**: qwen3.8 27B, gpt-oss 20B (low/high
  effort), qwen3.6 35B, qwen3 14B with thinking on.
