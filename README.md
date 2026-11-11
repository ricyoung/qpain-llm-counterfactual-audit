# Q-Pain LLM counterfactual audit

Code, data and paper for *Do Large Language Models Show Racial and Sex Bias in Pain Management?
A Counterfactual Audit of GPT, Claude, Gemini, Llama, DeepSeek and Qwen* (Richard J. Young, Alice M.
Matthews). Submitted to arXiv on 2026-11-10 (cs.CL; cross-listed to cs.AI and cs.CY); the arXiv
identifier will be added here once the paper is announced. Paper: `paper/main.pdf`; arXiv source: `paper/arxiv/`. Analysis: `experiment/`
(`analyze_panel.py`, `failure_by_identity.py`, `make_panel_assets.py`, `figures.py`; tests in `experiment/test_*.py`).

**About this copy.** Every request and response is saved under `experiment/outputs/raw*/`. In this
public copy the address of the local inference host is replaced by `study-host` in the captured
records, and absolute local paths are made relative; nothing else is changed. Q-Pain material
(`experiment/data/qpain/`, and the Q-Pain prompts inside the records) is redistributed under its
licence, CC BY-SA 4.0 (PhysioNet, doi:10.13026/61xq-mj56). To re-run: set `OLLAMA_HOST` for local
models and `OPENROUTER_KEY_FILE` (a file holding the key, mode 0600) for the API panel.

---

# It is unknown whether large language models reproduce known disparities in pain assessment.

Created 2026-11-05 from an Atlas idea card. Status: draft (2026-11-06).

**New session? Start with [HANDOFF.md](HANDOFF.md):** what is running, the remaining steps, tools and OpenRouter access.

## The gap
It is unknown whether large language models reproduce known disparities in pain assessment.

- Kind: open_problem; cluster: clinical
- Source: PAINT: A Framework for Multilingual Pain Representation from Medical Licensing Exams to Real-World Clinical Texts (arXiv:2610.18269), 2026-10-13
- Evidence: "We introduced PAINT, to our knowledge the first multilingual resource for pain representation across medical licensing examinations and real- world clinical text in English, Spanish, Portuguese, and Japanese. It contributes clinician-validated pain and demographic annotations and the open- weight labeling pipeline. PAINT is intended as a benchmark for multilin- gual clinical information extraction and as infras- tructure for studying how pain documentation, and the demographic detail attached to"

## Before committing
- [ ] `novelty_check` on the one-paragraph pitch (re-run within 7 days of submission)
- [ ] Hypothesis written as a falsifiable claim
- [ ] Models (open-weight first), data (public first) and compute fixed

## Prior work found by live search (2026-11-05) -- the gap is partly claimed
- *Pain* 2024, "Racial, ethnic, and sex bias in large language model opioid
  recommendations for pain management", doi:10.1097/j.pain.0000000000003388
- Q-Pain (2021), a QA dataset for social bias in pain management, doi:10.13026/61xq-mj56
- "Instability Floors: Separating Bias from Noise in Fairness Audits of Clinical
  LLM Agents" (arXiv:2609.03221, 2026)
- Omiye et al. 2023 (npj Digit Med); Zack et al. 2023 (Lancet Digit Health)

Read these before writing. An English-only race/sex vignette audit is **not novel**.

**Closest prior work, from its abstract (OpenAlex, 2026-11-05):** Young, Enichen,
Rao & Succi (*Pain* 2024) gave GPT-4 and Gemini 40 MIMIC-IV pain cases in every
race/ethnicity x sex combination (480 per model) and found that **race/ethnicity
and sex did not influence** pain ratings or opioid recommendations. Q-Pain (2021)
found significant intersectional gaps, but in GPT-2/GPT-3. Our v1/v2 nulls agree
with Young et al.; what we add is open-weight 2025-26 models, name-only cues (v3),
and the cross-lingual comparison, which neither tests.

## Plan (proposed angle: cross-lingual, noise-floored)
1. **Hypothesis:** demographic gaps in LLM pain severity and strong-opioid choice
   differ by prompt language (EN/ES/PT/JA, the PAINT languages), and exceed the
   model's own repeat-sampling noise floor in at least one language.
2. **Models:** open-weight, local on .202 (no provider quota): gemma3:12b,
   qwen3:14b, ministral-3:14b, llama3.2; add larger ones after the pilot.
3. **Data:** `experiment/vignettes.json`, 6 written cases × 8 race/sex arms; next,
   cases from PAINT's clinician-validated texts, translated and back-checked.
4. **Metrics and statistics:** paired difference against the White-man arm with the
   case and seed held fixed; bootstrap 95% CI; sign-flip permutation p; Holm
   correction across arms; noise floor = same-arm rep-to-rep variance.
5. **Threats to validity:** explicit demographic labels are not how bias enters
   real notes (add name-based and dialect cues); vignettes are author-written;
   temperature 0.7 sampling; JSON forcing may change answers.

## Status
- [x] Harness (`experiment/run.py`), analysis (`experiment/analyze.py`), 6 vignettes
- [x] Smoke test 2026-11-05: 48 calls, 0 errors, 0 unparseable (measured)
- [x] English pilot 2026-11-05, 4 models × 6 cases × 8 arms × 5 reps (measured:
  960 answers, 0 errors, 0 unparseable; `outputs/pilot-summary.json`).
  **Result: null.** Severity differences are exactly 0 for gemma3, qwen3 and
  llama3.2 and within ±0.07 for ministral-3; strong-opioid differences are at
  most +0.167 (ministral-3, Black man, p=0.13), none significant before
  correction. Mean severity 8.3–8.5, strong-opioid rate 0.47–0.53.
  **Design flaw:** every case states the patient's own rating ("Rates pain 9/10"),
  so models copy it; severity is anchored, not assessed.
- [x] Pilot v2 2026-11-05, same design, no self-reported rating (measured: 960
  answers, 0 errors, 0 unparseable; `outputs/v2-summary.json`). Models now assess:
  mean severity 7.1-7.7, range 5-9; strong-opioid rate 0.29 (gemma3) to 0.51.
  **Still null:** every severity gap within +/-0.10, every strong-opioid gap within
  +/-0.10, no p below 0.25. With explicit race/sex labels these four open models
  show no detectable disparity at n=30 pairs per arm (power is low: a 0.1-point
  gap is undetectable here).
- [x] Cross-lingual harness 2026-11-05: `experiment/languages.json` (es, pt, ja draft
  translations by Claude), `experiment/lang.py`, `run.py --lang`. Spanish smoke test:
  48 calls, 0 errors, 48 of 48 parsed (measured).
  **Before the main run:** native clinician review and back-translation; the case
  texts use masculine forms (es "quieto", "rigido"; pt "parado", "rigido") that do
  not agree for women patients.
- [x] Pilot v3 2026-11-05, name-only cues, 2 names per arm (measured: 1,920
  answers, 0 errors, 0 unparseable; `outputs/v3-summary.json`). **Null after
  correction.** gemma3 rates all 7 non-reference names +0.03 to +0.12 higher than
  the White-man names (largest Hispanic woman +0.12, p=0.015, Holm ~0.105):
  opposite in direction to the human disparity. **Per-name check: one name, not
  race** -- Greg Baker 7.67, Todd Walsh 7.77, all 16 names within 7.67-7.83. Others within +/-0.08; opioid gaps
  within +/-0.067.
- [x] Per-name breakdown 2026-11-05: gemma3's +0.03..+0.12 shift is the reference
  name Greg Baker (7.67, lowest of 16 names): vs Greg Baker mean diff +0.133,
  vs Todd Walsh +0.010 (measured, `experiment/make_paper_assets.py`).
- [x] Noise floor (`experiment/noise_floor.py`): strong-opioid identity flip rate
  at or below the same-prompt floor for all 8 model x condition sets; severity
  identity rate exceeds floor only for gemma3 (0.105 vs 0.083; 0.100 vs 0.071).
- [x] Minimum detectable paired difference (80% power): severity 0.08-0.17 points,
  strong-opioid 0.04-0.14 (llama3.2 severity: no variation at all).
- [x] Manuscript rewritten 2026-11-05 to the academic outline (abstract 226 words,
  third person, H1/H2 stated, related work critical, methods replicable, results
  descriptive, discussion P1..P7 + limitations + conclusion). Every number, table
  and figure regenerates from the raw responses via `experiment/make_paper_assets.py`
  -> `experiment/outputs/paper_stats.json`, `paper/tables/*.tex`, `paper/figures/*.pdf`.
  Builds with pdflatex+bibtex (14 pages). All 8 references resolve to DOI/arXiv.
- [x] 2026-11-05 evening, after Ric's read: moved to the house PRIMEarxiv template
  (house style reference files kept locally, not in this repository;
  `paper/main.tex` + `sections/01..08` + `references.bib`, unsrt). References
  grown from 8 to 56, every one verified against OpenAlex/arXiv (ids in the bib);
  found via Atlas search, live OpenAlex/S2/PubMed search and S2 citation walks of
  Young et al. 2025 and Salinas et al. 2024. Overview schematic added as Figure 1
  (`experiment/make_overview.py`). Authors: Richard J. Young (Lee Business School,
  Information Systems, UNLV; ryoung@unlv.nevada.edu) and Alice M. Matthews
  (DeepNeuro.ai). Ethics: no human subjects, IRB exempt (45 CFR 46.102).
  Competing interests: DeepNeuro.ai compute, owned by first author. 17 pages.
- [ ] Before submission: `sudo apt install cm-super` on the build machine so the typewriter
  font is Type 1 (arXiv has it anyway); confirm the study host's GPU line
  (two RTX A4000, from the 2026-09 topology note); read the full Young et al. 2024
  PDF so the methods comparison is exact; repository URL in Data availability.
- [ ] Powered cross-lingual run (needs native-speaker review of languages.json).
- [ ] (was) Pilot v3: add implicit cues (names, dialect)
  beside explicit labels; then the cross-lingual run
- [ ] Translations and the cross-lingual run
- [ ] Draft, `preprint-check`, Ric submits to arXiv

## Files
- `related_work.bib`: the source paper and the papers naming the same gap
- `experiment/`: harness stub
- `paper/main.tex`: preprint skeleton; cite only what you have read

## Licences

This repository uses several licences: OpenRAIL-S for the code, CC BY 4.0 for the paper and authored data, and CC BY-SA 4.0 for the Q-Pain material and everything adapted from it. See [LICENSE](LICENSE) for what covers what.
