# Reproducibility details

This file records the versions, backups and validation for the main evaluation. All main results use parser `explicit-rational-v3`. The model, prompts, seeds, voting rules and question manifests were kept unchanged when the saved responses were scored.

## Versions and backups

- Base commit: `f2571d547e4ea382e92c3fba5ace2f683f8e1f82`
- Source backup: `final-code-v3.zip`
- Source backup SHA-256: `e425dcb9d1ef6331250985a9937ae5007e42d473e38f6f5a8dae5f5a8aea6a80`
- Results backup: `eval100-v3-review.zip`; [download the published copy, `eval100-v3-review.3.zip`](https://github.com/KaneRowley/multi-agent-debate/releases/download/eval100-v3/eval100-v3-review.3.zip)
- Results backup SHA-256: `15cacc6663b339319b64c310835606ee98f7debe316d6f2c7c9b5d8cd0055d40`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model/tokenizer revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- 100 fixed test questions, three repeats, 6,000 records, 4,800 model calls and 1,200 reused records.
- The raw-response backup is attached to the [eval100-v3 release](https://github.com/KaneRowley/multi-agent-debate/releases/tag/eval100-v3). Its checksum is unchanged despite the download filename. The source backup is retained separately.

The repository includes the saved [summary](../results/eval100/summary.json), [question/round outcomes](../results/eval100/questions.json) and [run metadata](../results/eval100/run_metadata.json). These are unchanged copies from the results backup. Their original paths and SHA-256 checksums are recorded in the [export inventory](../results/PROVENANCE.json).

The [results guide](../results/README.md) also links the three summarisation pilot comparisons. To rerun answer extraction, [download the response ZIP](https://github.com/KaneRowley/multi-agent-debate/releases/download/eval100-v3/eval100-v3-review.3.zip) and restore its `eval100/` directory.

## Validation

The saved final evaluation implementation passed 28 offline unit tests. The later recorded suite passed 30 tests, including the summarisation follow-up checks. Parser regression tests are in `tests/test_scoring.py`.

Offline replay of all 6,000 saved response records reproduced all 1,800 v3 question/round outcomes, including predictions, voting status and correctness. These checks did not load the model or run new inference. Storage checks covered the Drive fallback, preserving existing files, cleanup and error handling.

The Drive fallback was tested with simulated failures. These tests do not establish how a live Drive mount behaves. Use one writer per run or analysis output directory, and only remove an `.active` lock after confirming that the previous process has stopped.

## Fresh-checkout verification (22 November 2026)

A fresh clone of commit `f0b3031ba6cd74d91b1406860273a6ff18c229b0` passed all 30 offline unit tests. The public `eval100-v3` release ZIP was downloaded and its SHA-256 matched the recorded backup checksum. After extraction, the documented `debate.analysis` command reproduced `summary.json` and `questions.json` exactly as JSON, including all 1,800 question/round outcomes. The [main accuracy table](../results/eval100/accuracy_table.md) was regenerated from that analysis.

These checks did not run the model or require a GPU. They verify offline software and saved-result reproduction; they do not constitute a new live test of model generation.

The completed [per-answer summarisation comparison](../results/summarisation/eval100-per-answer/comparison.json) contains 300 unique question-repeat outcomes. Its accuracy and token totals were independently checked against its rows. The privately retained run metadata was checked for per-answer mode, a 256-token summary limit, and the pinned T4/model runtime. The baseline metadata, source-response and reference hashes match the preserved main evaluation, and the paired question-bootstrap interval was independently reproduced.

## Reproduce scoring without inference

Download the response ZIP into the repository root and restore `eval100/`:

```bash
python -m zipfile -e eval100-v3-review.3.zip runs
```

Then run from the repository root:

```bash
python -m unittest discover -s tests -v
python -m debate.analysis --run runs/eval100 --references manifests/evaluation_references.json --output analysis/eval100-v3
```

Use a new analysis output directory and keep the original responses. Scoring does not require a GPU or model download.

## Final mean accuracy

| Configuration | Accuracy |
|---|---:|
| Single | 87.00% |
| Three-agent initial voting | 89.67% |
| Three agents, one revision | 91.33% |
| Three agents, two revisions | 91.67% |
| Five-agent initial voting | 92.67% |
| Five agents, one revision | 93.67% |

Three-agent debate after two revisions increased mean accuracy by 2.00 percentage points over its initial independent vote. The paired question-bootstrap 95% confidence interval is approximately -0.33 to +4.67 percentage points, leaving the additional benefit uncertain. All three repeats use the same 100 questions.
