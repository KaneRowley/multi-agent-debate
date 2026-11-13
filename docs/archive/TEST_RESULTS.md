# Offline validation results

Date: 11 November 2026. Python 3.12.1 on Windows; standard-library test suite.

Command: `python -m unittest discover -s tests -v`

All tests use mocked generation. No model weights were downloaded; no real inference or evaluation was run. These are software test outcomes, not benchmark results.

Additional verification: module compilation and CLI help succeeded; both saved manifests validate their IDs/text hashes and reference-key alignment; counts are 30 training and 100 test questions. The test selection has zero overlap with the documented 106 exclusions. All 1,319 old/new test rows and references matched in the exposure audit. `git diff --check` passed.

GPU compatibility, FP16 loading/generation, Colab dependency installation, actual runtime, explanation quality and benchmark accuracy remain unverified. Old reports/results were not modified. No commit or push was performed.

```text
test_external_initial_reuse_requires_exact_request (test_runner.RunnerTests.test_external_initial_reuse_requires_exact_request) ... ok
test_histories_snapshots_association_no_leakage (test_runner.RunnerTests.test_histories_snapshots_association_no_leakage) ... ok
test_manifest_selection_and_storage (test_runner.RunnerTests.test_manifest_selection_and_storage) ... ok
test_partial_summary_fixed_denominator (test_runner.RunnerTests.test_partial_summary_fixed_denominator) ... ok
test_repeat_seeds_and_round_counts (test_runner.RunnerTests.test_repeat_seeds_and_round_counts) ... ok
test_resume_preserves_records_and_reuses_initial (test_runner.RunnerTests.test_resume_preserves_records_and_reuses_initial) ... ok
test_summaries_cost_and_failures (test_runner.RunnerTests.test_summaries_cost_and_failures) ... ok
test_template_slots_do_not_modify_question_or_peer_text (test_runner.RunnerTests.test_template_slots_do_not_modify_question_or_peer_text) ... ok
test_ambiguity_conflicts_and_empty (test_scoring.ScoringTests.test_ambiguity_conflicts_and_empty) ... ok
test_exact_equivalence (test_scoring.ScoringTests.test_exact_equivalence) ... ok
test_markers_and_trailing_words (test_scoring.ScoringTests.test_markers_and_trailing_words) ... ok
test_plurality_and_labels (test_scoring.ScoringTests.test_plurality_and_labels) ... ok

----------------------------------------------------------------------
Ran 12 tests in 6.543s

OK

```
