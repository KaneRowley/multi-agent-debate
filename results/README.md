# Saved analysis results

These are exact JSON exports from the preserved experiment archives. No model
was run and no scores were changed when adding them to the repository.
[PROVENANCE.json](PROVENANCE.json) records the source archive hashes, original
member names and exported file hashes. Files named `run_metadata.json` are
unchanged copies of their source `run.json` files.

## Main evaluation

| File | Contents |
|---|---|
| [Summary](eval100/summary.json) | Final `explicit-rational-v3` analysis, per-seed accuracy, costs and primary bootstrap comparison |
| [Question outcomes](eval100/questions.json) | All 1,800 scored question/round outcomes, including predictions, correctness, extraction statuses and voting statuses |
| [Run metadata](eval100/run_metadata.json) | Fixed question selection, prompts, model revision, seed procedure and recorded runtime |

The six reported stages use the same 100 questions and three repeats. The
repeats are not 300 independent questions. Earlier scoring versions are not
included here; the report uses v3 throughout.

Each `per_round` entry reports both that round's costs and costs accumulated
through that round. Do not sum cumulative rows across rounds. The
`logical_input_tokens` and `logical_output_tokens` fields include the original
tokens of reused initial responses. The `input_tokens`, `output_tokens`,
`model_calls` and `generation_seconds` fields in the saved cost objects count
only newly generated work. As a result, the latter fields alone do not give
standalone configuration costs. Original measured durations for reused
responses can be recovered from the raw-response archive, as done for the
report's standalone time comparisons.

## Summarisation development pilots

All three variants use the same five training questions, one repeat, three
agents and one revision. They reuse the same initial answers and saved
full-response baseline. Each directory includes the saved comparison and run
metadata, including the source and code hashes.

| Variant | Results | Settings | Total tokens | Correct |
|---|---|---|---:|---:|
| Original per-recipient summaries | [Comparison](summarisation/dev-pilot/comparison.json) | [Metadata](summarisation/dev-pilot/run_metadata.json) | 37,855 | 5/5 |
| Revised per-recipient prompt | [Comparison](summarisation/dev-pilot-v2/comparison.json) | [Metadata](summarisation/dev-pilot-v2/run_metadata.json) | 36,724 | 5/5 |
| Reused per-answer summaries | [Comparison](summarisation/dev-pilot-per-answer/comparison.json) | [Metadata](summarisation/dev-pilot-per-answer/run_metadata.json) | 29,189 | 5/5 |

The shared full-response baseline used 30,611 tokens and answered 5/5
correctly. Token totals include initial generation, summaries and revisions.
Recorded times in these comparison files cover summaries and revisions only,
excluding the shared initial generation. The final variant used 4.65% fewer
tokens but its recorded summary/revision time was 32.76% longer. All five
initial group predictions were correct, so these exploratory development
results do not establish error-correction performance or general accuracy
preservation. The saved bootstrap interval of `[0, 0]` reflects identical
correctness on these five cases, not proof of equivalence on new questions.

## Raw responses and reproduction

The main raw generations are preserved in `eval100-v3-review.zip` (SHA-256 in
the provenance inventory). Summarisation requests, responses and checkpoints
are in the corresponding follow-up archive. These larger archives are not
committed here; the JSON exports let readers inspect the reported results,
but do not replace the raw responses for rerunning the parser.

After obtaining and extracting the evaluation archive, follow
[the offline scoring instructions](../docs/FINAL_EXPERIMENT.md#reproduce-scoring-without-inference).
Archived paths under `/content/drive/` describe the original Colab environment;
they are not locations that need to exist on a reader's machine.
