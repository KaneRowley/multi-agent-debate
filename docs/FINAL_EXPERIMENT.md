# Reproducibility details

This file records the versions, backups and validation for the main evaluation. All main results use parser `explicit-rational-v3`. The model, prompts, seeds, voting rules and question manifests were kept unchanged when the saved responses were scored.

## Versions and backups

- Base commit: `f2571d547e4ea382e92c3fba5ace2f683f8e1f82`
- Source backup: `final-code-v3.zip`
- Source backup SHA-256: `e425dcb9d1ef6331250985a9937ae5007e42d473e38f6f5a8dae5f5a8aea6a80`
- Results backup: `eval100-v3-review.zip`
- Results backup SHA-256: `15cacc6663b339319b64c310835606ee98f7debe316d6f2c7c9b5d8cd0055d40`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model/tokenizer revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- 100 fixed test questions, three repeats, 6,000 records, 4,800 model calls and 1,200 reused records.
- These ZIP backups are stored separately from the repository. They are needed to restore the complete code and raw responses.

The repository includes the saved [summary](../results/eval100/summary.json), [question/round outcomes](../results/eval100/questions.json) and [run metadata](../results/eval100/run_metadata.json). These are unchanged copies from the results backup. Their original paths and SHA-256 checksums are recorded in the [export inventory](../results/PROVENANCE.json).

The [results guide](../results/README.md) also links the three summarisation pilot comparisons. To rerun answer extraction, the original model responses from the separate ZIP backup are needed.

## Validation

The saved final evaluation implementation passed 28 offline unit tests. The later recorded suite passed 30 tests, including the summarisation follow-up checks. Parser regression tests are in `tests/test_scoring.py`.

Offline replay of all 6,000 saved response records reproduced all 1,800 v3 question/round outcomes, including predictions, voting status and correctness. These checks did not load the model or run new inference. Storage checks covered the Drive fallback, preserving existing files, cleanup and error handling.

The Drive fallback was tested with simulated failures. These tests do not establish how a live Drive mount behaves. Use one writer per run or analysis output directory, and only remove an `.active` lock after confirming that the previous process has stopped.

## Reproduce scoring without inference

With the archived `eval100` directory restored locally:

```bash
python -m unittest discover -s tests -v
python -m debate.analysis --run /path/to/eval100 --references manifests/evaluation_references.json --output /path/to/new-analysis-v3
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
