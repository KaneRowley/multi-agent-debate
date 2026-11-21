# Multi-agent debate on GSM8K

Does sharing worked solutions improve a language model's mathematical accuracy? This project compares a single agent, independent plurality voting, and multi-agent debate using **Qwen2.5-3B-Instruct** on **100 fixed GSM8K test questions**, with **three generation-seed repeats**.

**[Read the report (PDF, working draft)](docs/Multi_agent_Benchmark_Report_draft.pdf)** · [Inspect saved results](results/README.md) · [Score saved responses](#score-saved-responses-without-a-gpu) · [Run in Colab](#run-generation-in-colab)

**Status:** the main evaluation is complete. The PDF is the draft uploaded on 21 November 2026 and still needs updates; it is not a final report. The peer-summarisation follow-up is documented separately below.

## Main findings

Mean accuracy across three repeats, using parser `explicit-rational-v3`:

| Configuration | Zero revisions | One revision | Two revisions |
|---|---:|---:|---:|
| Single agent | 87.00% | — | — |
| Three agents | 89.67% | 91.33% | 91.67% |
| Five agents | 92.67% | 93.67% | — |

**Zero revisions means independent generation with plurality voting for multiple agents.** A revision lets each agent review the other agents' previous worked solutions while retaining its own conversation. A dash means that configuration was not evaluated.

The primary comparison, three agents after two revisions versus their initial vote, improves by **2.00 percentage points**. Its paired question-bootstrap 95% interval is approximately **−0.33 to +4.67 points**, so the additional benefit of debate is not established conclusively. The three repeats use the same 100 questions, not 300 independent questions.

Five-agent independent voting achieved higher mean accuracy than three-agent debate after two revisions (92.67% versus 91.67%), using about one-sixth as many cumulative input and output tokens per question. The configurations were not compared at matched token budgets.

## Results and report availability

| Artifact | Available here |
|---|---|
| Report | [Working-draft PDF](docs/Multi_agent_Benchmark_Report_draft.pdf), including methods, figures, detailed results and scoring appendices |
| Main evaluation | [Summary](results/eval100/summary.json), [question/round outcomes](results/eval100/questions.json) and [run metadata](results/eval100/run_metadata.json) |
| Summarisation development pilots | [Comparisons and settings for all three variants](results/README.md#summarisation-development-pilots) |
| Provenance | [Export inventory and checksums](results/PROVENANCE.json) and [final experiment record](docs/FINAL_EXPERIMENT.md) |
| Raw model responses | Preserved in separate backup archives; not bundled with the repository or available through a download link here |

The committed JSON files let readers inspect the reported outcomes and costs immediately. Replaying answer extraction requires the raw responses from `eval100-v3-review.zip`; the JSON analysis exports alone are insufficient. See the [results guide](results/README.md#raw-responses-and-reproduction) for the distinction between saved analyses and raw generations.

## Setup and offline checks

Clone the repository and use Python 3.11 or later. For the offline checks and scoring, the Python standard library is sufficient; a GPU and the Colab dependencies are not required.

```bash
git clone https://github.com/KaneRowley/multi-agent-debate.git
cd multi-agent-debate
```

Run the checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These standard-library tests do not download a model or run inference. The [validation record](docs/FINAL_EXPERIMENT.md#validation) distinguishes the preserved evaluation checks from the current follow-up suite. Archived offline replay reproduced all 1,800 saved v3 question/round outcomes.

## Score saved responses without a GPU

Restore `eval100/` from the results backup into `runs/eval100/`, then run from the repository root:

```bash
python -m debate.analysis --run runs/eval100 --references manifests/evaluation_references.json --output analysis/eval100-v3
```

Use a new analysis output directory if that destination exists. Check that the resulting `summary.json` reports `parser_version: explicit-rational-v3` and `complete: true`. Scoring reads saved responses and does not run the model.

## Run generation in Colab

Select a **T4 GPU**. In notebook cells:

```python
!git clone https://github.com/KaneRowley/multi-agent-debate.git /content/multi-agent-debate
%cd /content/multi-agent-debate
%pip uninstall -y torchvision torchaudio
%pip install -r requirements-colab.txt
```

Restart the Colab session after installation, then check the environment:

```python
%cd /content/multi-agent-debate
!python -m unittest discover -s tests -v
!nvidia-smi
```

A small development run, using the first ten questions and all configurations:

```python
!python -u -m debate.runner --manifest manifests/development.json --limit 10 --repeats 1 --output runs/dev-smoke
!python -m debate.analysis --run runs/dev-smoke --references manifests/development_references.json --output analysis/dev-smoke-v3
```

For a reproduction of the frozen evaluation, mount Drive so completed records survive runtime deletion. Use a new output folder; the example uses `eval100-reproduction` to keep it separate from the saved original:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!python -u -m debate.runner --manifest manifests/evaluation.json --repeats 3 --output /content/drive/MyDrive/sakana-results/eval100-reproduction --evaluation-ready
```

The first generation run downloads model weights. `--evaluation-ready` explicitly acknowledges that the design is frozen; the command above reproduces the existing experiment rather than requesting a new design search.

To continue a stopped run, repeat its exact command with `--resume` appended. Restore the same code, GPU/runtime settings and Drive mount first. Resume checks compatibility and skips completed records. Run only one writer per output directory. If a hard interruption leaves `.active`, confirm the old process has ended before removing that lock. Never delete completed response files to obtain different answers.

Local `/content` storage is temporary. Keep both the final code and results ZIP backups outside it. Historical smoke-to-development continuation instructions are retained under [the archive](docs/archive/README.md); their v2 instructions describe an earlier stage, not the final scoring setup.

## Method

- **Model:** `Qwen/Qwen2.5-3B-Instruct`, pinned model/tokenizer revision; Transformers/PyTorch, FP16 on a Tesla T4.
- **Generation:** temperature 0.7, top-p 0.9, top-k 0, at most 1,024 new tokens.
- **Development:** 30 fixed GSM8K training questions. Evaluation uses 100 fixed test questions and seed roots 101, 202 and 303.
- **Debate:** independent initial answers, then synchronous previous-round peer sharing. No judge, calculator, role specialization or agreement-based early stopping.
- **Scoring:** explicit numerical extraction, exact rational equivalence, and plurality voting. Invalid answers do not vote; tied votes abstain and score incorrect. Format compliance is measured separately.
- **Selection:** evaluation IDs exclude 106 documented prior-exposure IDs. Additional download-only exposure and model pretraining contamination remain limitations; the sample is not claimed to be entirely untouched.

This adapts [Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325). Model, prompts, selection, scale and scoring differ from the published experiment. The [final experiment record](docs/FINAL_EXPERIMENT.md) gives the model pin, archive checksums and validation details.

## Recorded experiment cost

The main evaluation contains 6,000 saved response records: 4,800 actual model calls and 1,200 exact-request reuses. Recorded generation time totals about 19.53 T4 GPU-hours, excluding startup, file I/O and interruptions. This is the combined experiment cost, not the standalone cost of any one configuration. Standalone token comparisons include the original token usage of reused responses; see the [results guide](results/README.md) and the report's detailed cost table.

## Separate follow-up: peer summarisation

The [three development pilots](results/README.md#summarisation-development-pilots) use five training questions and one repeat. All variants scored 5/5; reused per-answer summaries used 4.65% fewer total tokens but took 32.76% longer for summaries and revisions. These small development results do not establish accuracy preservation. Full 100-question follow-up results are not included in this repository yet and are separate from the main results above.

The commands below use the default **per-recipient** summary procedure. To reproduce the per-answer variant, add `--summary-mode per-answer` and use a separate output folder. Preserve the chosen mode, token limit and other options when resuming.

Use the T4 setup above and mount Drive. Set `baseline` to the Drive folder containing the saved `dev30/` and `eval100/` run directories, each with `run.json` and `generations/`. Restore `runs/dev30/` from `dev30-review.zip` and `eval100/` from `eval100-v3-review.zip` if needed. Keep these runs in place. The follow-up compares one three-agent revision using summarised peer responses with the saved first revision, scored with `explicit-rational-v3`.

```python
baseline = "/content/drive/MyDrive/sakana-results"
followup = "/content/drive/MyDrive/sakana-summarisation-followup"
```

Run the five-question development pilot first, then the evaluation on the saved 100 questions and all three repeats:

```python
!python -u -m debate.summarisation_followup --source "{baseline}/dev30" --references manifests/development_references.json --dev-limit 5 --output "{followup}/dev-pilot"
!python -u -m debate.summarisation_followup --source "{baseline}/eval100" --references manifests/evaluation_references.json --output "{followup}/eval100"
```

Resume with the same code, runtime, paths and options. For evaluation:

```python
!python -u -m debate.summarisation_followup --source "{baseline}/eval100" --references manifests/evaluation_references.json --output "{followup}/eval100" --resume
```

For the pilot, append `--resume` to its command. If a hard stop leaves `.active`, add `--clear-stale-lock` only after confirming the previous process has ended. Use one writer per output directory. Saved summaries and revisions are reused, and `comparison.json` is written after completion. Keep run files and checkpoints on Drive.

## Repository layout

| Location | Purpose |
|---|---|
| `debate/` | Dataset selection, prompts, generation, runner, scoring and analysis |
| `manifests/` | Fixed questions and separate evaluator references |
| `tests/` | Offline software tests and response fixtures |
| `results/` | Saved evaluation analyses, metadata and summarisation pilot comparisons |
| `docs/` | Draft PDF report, final experiment record and original design plan |
| `docs/archive/` | Historical development instructions and audits |
| `prior_test_exposure.json` | Exposure inventory used by dataset selection |
| `requirements-colab.txt` | Pinned GPU runtime dependencies |

## Development history

The [documentation index](docs/README.md) separates the final experiment record from the [original design plan](docs/EXPERIMENT_PLAN.md) and [development archive](docs/archive/README.md). The plan and archived v1/v2 instructions record earlier stages; use the current instructions above for final v3 scoring. Development pilots are labelled separately from the 100-question main evaluation.
