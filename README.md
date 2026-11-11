# GSM8K debate experiment

Small, direct implementation of the approved plan in `EXPERIMENT_PLAN.md`. No benchmark results are included or claimed. The old `lllm-multiagent-debate-benchmark` repository is read-only source material for the exposure audit; this repository does not modify it.

## Components and flow

1. `debate/dataset.py` pins GSM8K, excludes known exposed test IDs, and writes ordered question manifests plus separate evaluator references. Saved manifests are already included: 30 development questions and 100 evaluation questions, disjoint from 106 documented test exclusions.
2. `debate/prompts.py` builds the exact worked-solution prompts. `debate/generation.py` checks one CUDA GPU and FP16, loads pinned Qwen/tokenizer, and logs effective decoding and hardware settings.
3. `debate/runner.py` gives each agent a private conversation. Round 0 is independent. Revisions append peers' completed previous responses to each private history. Requests for a round are built before generating any revision. The single/three/five-agent trajectories save 1/9/10 responses per question.
4. `debate/scoring.py` extracts explicit numerical answers as exact fractions and votes by plurality. `debate/analysis.py` alone reads reference labels, scores every selected question, and writes per-round/cumulative costs and uncertainty. It never adds reference answers to prompts.

Only initial responses with matching request fingerprints are reused. There are 16 actual calls per question/repeat with shared initials, versus 20 saved logical responses. Records say where reused responses came from. Numerical accuracy and compliance with the requested final-box format are distinct metrics.

## Offline validation

Python 3.11 or 3.12; no third-party packages or model downloads required:

```bash
python -m unittest discover -s tests -v
```

Tests use a mocked generator and temporary directories. They check histories, synchronous snapshots, question/agent association, absence of labels in prompts, parsing/equivalence/conflicts/ties, revisions, repeated seeds, resume, initial reuse, incremental durability, fixed denominators and compute accounting. They are not inference results.

## Exact Colab setup

The working implementation is not automatically committed or pushed. Use the delivered `multi-agent-debate-colab.zip` snapshot. In a new Colab notebook select **Runtime → Change runtime type → T4 GPU**. Upload and unpack in a Python cell:

```python
from google.colab import files
import zipfile
uploaded = files.upload()  # choose multi-agent-debate-colab.zip
assert "multi-agent-debate-colab.zip" in uploaded
with zipfile.ZipFile("multi-agent-debate-colab.zip") as archive:
    archive.extractall("/content")
```

In subsequent cells:

```python
%cd /content/multi-agent-debate
%pip uninstall -y torchvision torchaudio
%pip install -r requirements-colab.txt
```

The unused vision/audio packages are removed to avoid conflicts with Colab's preinstalled Torch stack. Restart the Colab session after installation (**Runtime → Restart session**), then run:

```python
%cd /content/multi-agent-debate
!python -m unittest discover -s tests -v
!nvidia-smi
```

The first real run below will download the pinned model/tokenizer. Nothing in the offline tests does that. Hardware is checked before loading: one visible CUDA GPU, compute capability at least 7.0, and a finite FP16 matrix operation. Model weights are FP16 on T4; BF16 is never assumed. Pins in `requirements-colab.txt` are the intended Colab environment, not a claim that this environment has been GPU-tested. Resolve installation incompatibilities before inference and record any package changes.

## Development commands

Use the included manifests without regenerating them. The first **10-question smoke check**, all approved configurations, one development repeat:

```python
!python -m debate.runner --manifest manifests/development.json --limit 10 --repeats 1 --output runs/dev-smoke
```

Score it separately:

```python
!python -m debate.analysis --run runs/dev-smoke --references manifests/development_references.json --output analysis/dev-smoke
```

Inspect raw worked explanations, format/validity rates, truncation, memory and `projected_test_3_repeats_gpu_hours`. Review the proposed feasibility criteria in the plan. The projection includes distinct workload costs, including reused initials; it is not a guaranteed wall-clock budget.

After reviewing smoke, run all 30 development questions. Reuse matching completed smoke initial generations, preserving provenance:

```python
!python -m debate.runner --manifest manifests/development.json --limit 30 --repeats 1 --reuse-initial runs/dev-smoke --output runs/dev30
!python -m debate.analysis --run runs/dev30 --references manifests/development_references.json --output analysis/dev30
```

If interrupted, repeat the **same** runner command with `--resume` appended. Existing completed generations are verified and replayed, never overwritten. Reuse inputs must remain available. Changing the manifest, prompt, source code or runtime requires a new run directory. The runner stops on backend exceptions; it does not automatically retry errors. EOS/length/empty completions are final records. Context overflow is saved with an empty response and no model call. A hard process kill may leave `.active`; verify no writer remains and inspect partial files before manually removing that lock. Do not remove generation files to fish for a better answer.

Analysis is also append-only by directory: use a fresh analysis output name for each partial/final snapshot. The selected-question denominator stays fixed even for a partial run. Per-round/cumulative `model_calls`, tokens and seconds charge only newly generated records in that run; `logical_*` fields include reused response exposure. Global totals sum records once, not round summaries. Runtime projection counts distinct request fingerprints, including externally reused initials at their original measured duration. It excludes model startup and may underestimate long evaluation prompts.

Colab `/content` is temporary. Download run directories and analysis before the session is discarded, and restore them before resuming. Preserve source snapshots, manifests and references alongside them. No main evaluation is launched by these commands.

## Selection, audit and evaluation readiness

`prior_test_exposure.json` contains the recovered old 100 IDs, source commits/blob hashes, the row-mapping check and extra paper/audit exclusions. All 1,319 old/new test rows match. The new evaluation IDs exclude the full 106-ID union. Colab/download-only files have not all been checked, so do not describe the sample as untouched. No additional file is needed for the recovered 100; provide any result file with a different test `question_selection` or `question_id` set for a further audit.

Only if a reviewed exclusion inventory changes, create **new** manifests (never overwrite the included fixed selection):

```bash
python -m debate.dataset --exposure prior_test_exposure.json --output manifests-reviewed
```

This downloads dataset files only, pins their revision and records file checksums. It does not load models or select on answers. The default included manifest files were created before inference. Generation reads only the label-free manifest; `_references.json` files are evaluator inputs.

Main evaluation is intentionally gated by `--evaluation-ready`. Only after development runtime/explanation review, exposure review, and final design freeze should a future evaluation invocation use `manifests/evaluation.json`, `--repeats 3`, a fresh output directory, and that flag. This README does not launch evaluation or report hypothetical scores.

## Still unverified

- Pinned Colab package installation, actual T4 model loading, FP16 generation stability and peak memory.
- Runtime/token cost, usable worked explanations, single-agent capability and the three-repeat budget.
- Debate accuracy, scaling behavior and statistical uncertainty on real outputs.
- Additional exposure in Colab/download-only result files and pretraining contamination.

The first inference task is the 10-question development smoke command above. No real inference, model downloads, commits, pushes or edits to old report results were performed during implementation.
