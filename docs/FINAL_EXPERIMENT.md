# Preserved evaluation version

The final evaluation uses `explicit-rational-v3`, recovered from the working Colab source backup. The parser and storage implementation are preserved without new inference or changes to the model, prompts, seeds, voting, or manifests.

## Provenance

- Base commit: `f2571d547e4ea382e92c3fba5ace2f683f8e1f82`
- Source backup: `final-code-v3.zip`
- Source backup SHA-256: `e425dcb9d1ef6331250985a9937ae5007e42d473e38f6f5a8dae5f5a8aea6a80`
- Results backup: `eval100-v3-review.zip`
- Results backup SHA-256: `15cacc6663b339319b64c310835606ee98f7debe316d6f2c7c9b5d8cd0055d40`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model/tokenizer revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- 100 fixed test questions, three repeats, 6,000 records, 4,800 model calls and 1,200 reused records.
- These archives are separate backups, not bundled into this repository change. Keep both in persistent storage and a second downloaded copy.

The final [summary](../results/eval100/summary.json),
[question/round outcomes](../results/eval100/questions.json) and
[run metadata](../results/eval100/run_metadata.json) are now included in the
repository as exact exports from the results backup. Their source paths and
SHA-256 checksums are recorded in [the export inventory](../results/PROVENANCE.json).
The [results guide](../results/README.md) also links all three development
summarisation comparisons. These analysis exports do not contain the raw model
responses; the separate archive is still required to replay answer extraction.

## Validation

The preserved final evaluation implementation passed 28 offline unit tests. The current suite passes 30 tests, including the summarisation follow-up checks; parser regression tests are grouped in `tests/test_scoring.py`. Offline replay of all 6,000 saved records reproduces all 1,800 v3 question/round outcomes exactly, including predictions, vote statuses, and correctness. No model was loaded and no inference ran. One continuation test's obsolete expected parser label was updated from v2 to v3. New regression tests cover bounded parser introductions, conflicting answers, Drive fallback, existing-file preservation, cleanup, and error propagation.

Drive fallback is simulated in the storage unit tests; no live Google Drive mount is available here. The preserved fallback assumes a single writer holding the run lock. Never remove `.active` while another process is writing; analysis into a given output directory must also have only one writer. Drive rename/durability behavior is provided by the mount and is not claimed to be a cross-client atomic no-overwrite guarantee.

## Reproduce scoring without inference

With the archived `eval100` directory restored locally:

```bash
python -m unittest discover -s tests -v
python -m debate.analysis --run /path/to/eval100 --references manifests/evaluation_references.json --output /path/to/new-analysis-v3
```

Use a new analysis output directory; preserve the original responses and v2/v3 analyses. No GPU or model download is needed for scoring.

## Final mean accuracy

| Configuration | Accuracy |
|---|---:|
| Single | 87.00% |
| Three-agent initial voting | 89.67% |
| Three agents, one revision | 91.33% |
| Three agents, two revisions | 91.67% |
| Five-agent initial voting | 92.67% |
| Five agents, one revision | 93.67% |

Three-agent revision 2 minus initial voting is +2.00 percentage points; the saved question-bootstrap 95% interval is approximately -0.33 to +4.67 points. Repeats share the same 100 questions and do not represent 300 independent questions.
