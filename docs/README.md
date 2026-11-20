# Documentation

Start with the [project overview and reproduction commands](../README.md).

| Document | Status and purpose |
|---|---|
| [Final experiment](FINAL_EXPERIMENT.md) | Current v3 results, exact model pin, archive checksums, reproduction and validation |
| [Saved analysis results](../results/README.md) | Final evaluation summaries, question-level outcomes and all three summarisation pilot variants |
| [Experiment plan](EXPERIMENT_PLAN.md) | Original design approved before evaluation; retained as a historical record |
| [Development archive](archive/README.md) | Earlier implementation checks and smoke-to-development continuation evidence |

The original plan refers to parser v1 and work that had not yet run when it was written. The final experiment record identifies parser v3 as the implemented scoring version. Generation settings, prompts, manifests and voting were preserved. Current commands are in the main README; archived v1/v2 commands are not the instructions for final scoring.

The exposure inventory remains at [`prior_test_exposure.json`](../prior_test_exposure.json), supplied through `--exposure` when preparing manifests. Executable files and manifests stay in their original locations. Historical audit files are kept in the archive with their contents unchanged.
