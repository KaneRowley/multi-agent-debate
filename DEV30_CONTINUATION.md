# Continue the completed smoke run to 30 development questions

Preparation is complete locally: the separate `runs/dev30` contains 200 inherited response records (all smoke initials and revisions), with 400 response records still missing for the remaining 20 questions. No inference has been run for those questions. The existing 30-question manifest is unchanged. The original smoke directory and earlier analyses remain unchanged.

`--continue-from` requires a completed prefix run. It reconstructs every saved conversation and previous-round peer message, verifies request fingerprints and the full model/runtime/settings identity, and records the original run/record hashes and reuse chain. It retains completed empty/length/overflow responses too; parsing and correctness never decide reuse eligibility. A mismatch stops continuation rather than regenerating smoke responses. A separate output directory protects the source.

Parser/analysis source hashes are audit evidence, not inference-compatibility gates. Prompts, settings, seed rules, the fixed manifest and actual request/runtime identities still must match. Parser-only edits therefore permit replay without inference. The parser remains `explicit-rational-v2`; prompts, model, generation settings and voting are unchanged.

## Colab commands

Continue in the same Colab runtime/environment used for smoke. Do not upgrade or reinstall packages. If starting a new session, restore the original `runs/dev-smoke` directory and the same runtime identity first; incompatible runtime identity will fail before missing inference is generated.

Upload `dev30-continuation-update.zip` and apply it at the existing repository root:

```python
%cd /content/multi-agent-debate
from google.colab import files
from pathlib import Path
import zipfile

uploaded = files.upload()  # choose dev30-continuation-update.zip
assert "dev30-continuation-update.zip" in uploaded
with zipfile.ZipFile("dev30-continuation-update.zip") as archive:
    archive.extractall(".")
assert Path("runs/dev-smoke/run.json").is_file(), "Restore the original smoke directory first"
```

The update contains no `runs/` directory, model weights, prompt file, model/configuration file or old analysis. It leaves saved smoke work in place. If the source exists only in `dev-smoke-review.zip`, upload that archive and restore just its `runs/dev-smoke/` members into the repository root before continuing; never overwrite an existing source directory.

Validate software and prepare a new destination, without loading a model:

```python
!python -c "from debate.scoring import PARSER_VERSION; assert PARSER_VERSION == 'explicit-rational-v2'"
!python -m unittest discover -s tests -p test_continuation.py -v
!python -m debate.runner --manifest manifests/development.json --limit 30 --repeats 1 --continue-from runs/dev-smoke --output runs/dev30 --prepare-only
```

Expected preparation counts: `selected_questions=30`, `expected_responses=600`, `saved_responses=200`, `missing_responses=400`, `source_responses=200`. If `runs/dev30` is already prepared, add `--resume` to preparation rather than deleting it.

When ready, the following command performs **only missing inference**. It is provided for you to run; it has not been launched during preparation:

```python
!python -m debate.runner --manifest manifests/development.json --limit 30 --repeats 1 --continue-from runs/dev-smoke --output runs/dev30 --resume
```

Repeat that exact resume command if interrupted. Keep the source smoke directory unchanged and available. Source record identities are based on content, so a restored copy may have a different directory path; hashes and requests still must match. The smoke recorded Tesla T4/FP16, CUDA 12.4, Python 3.13.16, Torch 2.6.0, Transformers 4.51.3, Tokenizers 0.21.1, huggingface-hub 0.30.2 and safetensors 0.5.3. Actual runtime compatibility is checked when loading the generator, beyond the saved-identity checks performed during offline preparation.

After the remaining work completes, analyze **all 30 questions** afresh under parser v2 in a new directory:

```python
!python -m debate.analysis --run runs/dev30 --references manifests/development_references.json --output analysis/dev30-parser-v2
```

Verify completion without rerunning inference:

```python
import json
from pathlib import Path
summary = json.loads(Path("analysis/dev30-parser-v2/summary.json").read_text())
assert summary["parser_version"] == "explicit-rational-v2"
assert summary["complete"]
assert all(row["questions"] == row["completed_questions"] == 30 for row in summary["per_round"])
```

Use a new analysis destination if that name already exists. Do not merge old v1 smoke scores with v2 scores. The analysis reads the inherited raw responses alongside new responses and applies identical v2 extraction and voting to all of them. Preparation alone cannot supply complete 30-question scores; those await the missing inference.

## Costs and validation

The remaining 20 questions normally require 320 new model calls. A completed continuation has 600 logical response records: 200 inherited and 400 newly materialized, with 80 of the latter sharing initial answers. `actual_total_cost` and per-round/cumulative cost fields charge new compute only. `carried_forward_cost` preserves the original 160-call smoke workload once per request. `distinct_workload_cost_including_reused` accounts for the full 480-call workload, without double-counting shared initial records. All original token counts, durations and finish reasons are preserved in copied results. Context overflows may reduce actual model calls and remain failures; they are never silently repaired.

All 22 offline tests pass, including the five focused continuation tests. The 10-to-30 mock test proves that no initial or revision request from the first ten reaches the generator, verifies source/imported files and results, exercises prepared-run resume, scores all 30 with parser v2, and checks 320 new / 160 inherited / 480 total calls. Further tests cover parser-only resume, mismatched identities/settings/prompts, altered histories, incomplete sources, completed empty responses and source-directory protection. These are software tests, not new experiment results.

No prompts, model/settings or voting changes; no new inference, commits or pushes were performed while preparing this continuation.
