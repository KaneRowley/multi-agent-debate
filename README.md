# Multi-agent debate on GSM8K

Multi-agent debate gives language models a chance to catch each other's mistakes. This project investigates whether increasing the number of agents or revision rounds improves mathematical accuracy. We compare a single agent, independent plurality voting and multi-agent debate using **Qwen2.5-3B-Instruct** on **100 fixed GSM8K test questions**, with **three generation-seed repeats**.

**[Read the report (PDF)](docs/Multi_agent_Benchmark_Report.pdf)** · [Inspect saved results](results/README.md) · [Score saved responses](#score-saved-responses-without-a-gpu) · [Run in Colab](#run-generation-in-colab)

The main evaluation and per-answer summarisation follow-up are complete. The report was updated on 22 November 2026 and includes both. The saved outputs and offline reproduction instructions are linked below.

## Main findings

Mean accuracy across three repeats, using parser `explicit-rational-v3`:

| Configuration | Zero revisions | One revision | Two revisions |
|---|---:|---:|---:|
| Single agent | 87.00% | — | — |
| Three agents | 89.67% | 91.33% | 91.67% |
| Five agents | 92.67% | 93.67% | — |

**Zero revisions means independent generation with plurality voting for multiple agents.** A revision lets each agent review the other agents' previous worked solutions while retaining its own conversation. A dash means that configuration was not evaluated.

Our primary comparison is three-agent debate after two revisions against its initial independent vote. Mean accuracy increased by **2.00 percentage points**, with a paired question-bootstrap 95% confidence interval of **−0.33 to +4.67 percentage points**. This leaves the additional benefit of debate uncertain. All three repeats use the same 100 questions.

Five-agent independent voting achieved slightly higher mean accuracy than three-agent debate after two revisions (92.67% versus 91.67%), using approximately one-sixth as many tokens. Among the configurations tested, it offered a better balance between accuracy and token usage. A comparison at matched token budgets would help assess whether debate is a better use of the additional computation.

## Results and report availability

| File or results | Where to find them |
|---|---|
| Report | [Report PDF](docs/Multi_agent_Benchmark_Report.pdf), including methods, figures, detailed results and scoring appendices |
| Main evaluation | [Accuracy table](results/eval100/accuracy_table.md), [summary](results/eval100/summary.json), [question/round outcomes](results/eval100/questions.json) and [run metadata](results/eval100/run_metadata.json) |
| Summarisation evaluation | [Comparison and all 300 outcomes](results/summarisation/eval100-per-answer/comparison.json), with [settings and per-seed results](results/README.md#summarisation-evaluation) |
| Summarisation development pilots | [Comparisons and settings for all three variants](results/README.md#summarisation-development-pilots) |
| Provenance | [Export inventory and checksums](results/PROVENANCE.json) and [final experiment record](docs/FINAL_EXPERIMENT.md) |
| Raw model responses | [Download the main evaluation ZIP](https://github.com/KaneRowley/multi-agent-debate/releases/download/eval100-v3/eval100-v3-review.3.zip) from the [eval100-v3 release](https://github.com/KaneRowley/multi-agent-debate/releases/tag/eval100-v3) |

The JSON files contain the reported scores, predictions and costs. To rerun answer extraction, [download the response ZIP](https://github.com/KaneRowley/multi-agent-debate/releases/download/eval100-v3/eval100-v3-review.3.zip). Its SHA-256 matches the preserved evaluation backup. See the [results guide](results/README.md#raw-responses-and-reproduction) for the checksum and extraction details.

## Setup and offline checks

Use Python 3.11 or later and clone the repository. The offline tests and scoring use the Python standard library and do not require a GPU.

```bash
git clone https://github.com/KaneRowley/multi-agent-debate.git
cd multi-agent-debate
```

Run the checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests do not download or run the model. An offline replay reproduced all 1,800 saved v3 question/round outcomes. The [validation record](docs/FINAL_EXPERIMENT.md#validation) gives further details.

## Score saved responses without a GPU

Download [`eval100-v3-review.3.zip`](https://github.com/KaneRowley/multi-agent-debate/releases/download/eval100-v3/eval100-v3-review.3.zip) into the repository root. Extract it and score the saved responses:

```bash
python -m zipfile -e eval100-v3-review.3.zip runs
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
results_folder = input("Enter a new results folder on your mounted Drive: ").rstrip("/")
```

```python
!python -u -m debate.runner --manifest manifests/evaluation.json --repeats 3 --output "{results_folder}" --evaluation-ready
```

The model weights are downloaded on the first run. `--evaluation-ready` confirms that the experimental setup has been fixed for evaluation.

To continue a stopped run, repeat its exact command with `--resume` appended. Restore the same code, GPU/runtime settings and Drive mount first. Resume checks compatibility and skips completed records. Run only one writer per output directory. If a hard interruption leaves `.active`, confirm the old process has ended before removing that lock. Never delete completed response files to obtain different answers.

Files under `/content` are temporary. Keep the code and results backups on Drive or download a separate copy.

## Method

- **Model:** `Qwen/Qwen2.5-3B-Instruct`, pinned model/tokenizer revision; Transformers/PyTorch, FP16 on a Tesla T4.
- **Generation:** temperature 0.7, top-p 0.9, top-k 0, at most 1,024 new tokens.
- **Development:** 30 fixed GSM8K training questions. Evaluation uses 100 fixed test questions and seed roots 101, 202 and 303.
- **Debate:** agents first answer independently, then revise using the other agents' responses from the previous round. Each agent keeps its own conversation history. All scheduled rounds are completed, even if agents agree. No judge or calculator is used.
- **Scoring:** explicit numerical extraction, exact rational equivalence, and plurality voting. Invalid answers do not vote; tied votes abstain and score incorrect. Format compliance is measured separately.
- **Question selection:** 106 documented previously exposed test questions are excluded before selection. Questions are ordered by a deterministic hash and the first 100 eligible IDs are selected. This does not rule out other exposure or overlap with the model's pretraining data.

This experiment is based on [Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325), using a different model, question selection, prompts and scoring setup. The [final experiment record](docs/FINAL_EXPERIMENT.md) gives the exact versions, backup checksums and validation details.

## Recorded experiment cost

The main evaluation saved 6,000 response records. Of these, 4,800 came from model calls and 1,200 reused responses to identical initial requests. Total recorded generation time was approximately 19.53 hours on a T4 GPU, excluding model loading, file I/O and interruptions. This measures the combined experiment workload. When comparing individual configurations, we include the original cost of any reused responses; see the [results guide](results/README.md) and the report's cost table.

## Summarisation follow-up

Revision rounds substantially increased token usage, which motivated a [summarisation pilot](results/README.md#summarisation-development-pilots) on five development questions and one repeat. Summarising each initial answer once and reusing it saved 4.65% of total tokens in that pilot, but its summary and revision time was 32.76% longer. All five initial group predictions were already correct.

We then evaluated this **per-answer** procedure on the same 100 test questions and all three repeats. Each agent retains its own original solution and receives summaries of the other two agents' initial responses. Summaries have a 256-token output limit. The comparison uses the saved first full-response revisions from the three-agent main evaluation.

| Measurement | Full-response debate | Per-answer summarised debate |
|---|---:|---:|
| Mean accuracy | 91.33% (274/300) | 89.67% (269/300) |
| Mean total tokens per question-repeat outcome | 5,934.35 | 5,774.80 |
| Mean summary and revision time (s) | 39.10 | 51.74 |
| Invalid revision predictions | 16/900 | 75/900 |
| Revision responses stopped by the token limit | 0 | 0 |

Total tokens include the original initial generation, summaries where applicable, and revisions. Times exclude shared initial generation, model loading, checkpoint I/O and interruptions; they are summed recorded request times. No summary reached its token limit.

Total token usage fell by **2.69%**, while recorded summary and revision time increased by **32.33%**. The accuracy difference was **−1.67 percentage points** (paired question-bootstrap 95% CI: **−4.00 to +0.33 points**). This does not establish an accuracy reduction or that accuracy was preserved. The observed small token saving came with longer generation time and more parsing failures.

The [comparison file](results/summarisation/eval100-per-answer/comparison.json) contains every question-repeat outcome and the aggregate measurements. This used the same pinned model and T4 runtime as the main evaluation, per-answer summary mode, a 256-token summary limit and the same question selection. Follow-up raw summary/revision checkpoints and the original run metadata are retained privately and are not included in the public main-evaluation ZIP.

### Generate a new summarisation evaluation

These commands generate new summaries and revisions and require the T4 setup above. Reading the published comparison does not require a GPU. Download and extract the main-evaluation response ZIP as described above, then mount Drive and set a new follow-up output path:

```python
from google.colab import drive
drive.mount("/content/drive")

baseline = "/content/multi-agent-debate/runs"
followup = input("Enter a new follow-up folder on your mounted Drive: ").rstrip("/")
```

Use the explicit `--summary-mode per-answer` option to select the evaluated method:

```python
!python -u -m debate.summarisation_followup --source "{baseline}/eval100" --references manifests/evaluation_references.json --summary-mode per-answer --output "{followup}/eval100-per-answer"
```

To resume that run, keep the same code, runtime, paths and options:

```python
!python -u -m debate.summarisation_followup --source "{baseline}/eval100" --references manifests/evaluation_references.json --summary-mode per-answer --output "{followup}/eval100-per-answer" --resume
```

The saved source path and file hashes are checked on resume. If a hard stop leaves `.active`, add `--clear-stale-lock` only after confirming the previous process has ended. Use one writer per output directory. Completed summaries and revisions are reused; only missing requests are generated. Keep checkpoints on Drive, and retain the local source archive so it can be restored at the same path after a runtime reset. `comparison.json` is written after completion.

The earlier per-recipient pilots are documented in the [results guide](results/README.md#summarisation-development-pilots).

## Repository layout

| Location | Purpose |
|---|---|
| `debate/` | Dataset selection, prompts, generation, runner, scoring and analysis |
| `manifests/` | Fixed questions and separate evaluator references |
| `tests/` | Offline software tests and response fixtures |
| `results/` | Main evaluation analyses, regenerated accuracy table, and summarisation pilot/evaluation comparisons |
| `docs/` | Report and reproducibility details |
| `prior_test_exposure.json` | Exposure inventory used by dataset selection |
| `requirements-colab.txt` | Pinned GPU runtime dependencies |

## AI assistance

ChatGPT, Codex and Claude were used to help with implementation, debugging, repository documentation, and drafting or editing parts of the report.
