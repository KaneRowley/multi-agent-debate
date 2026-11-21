# Multi-agent debate on GSM8K

Multi-agent debate gives language models a chance to catch each other's mistakes. This project investigates whether increasing the number of agents or revision rounds improves mathematical accuracy. We compare a single agent, independent plurality voting and multi-agent debate using **Qwen2.5-3B-Instruct** on **100 fixed GSM8K test questions**, with **three generation-seed repeats**.

**[Read the report (PDF, working draft)](docs/Multi_agent_Benchmark_Report_draft.pdf)** · [Inspect saved results](results/README.md) · [Score saved responses](#score-saved-responses-without-a-gpu) · [Run in Colab](#run-generation-in-colab)

The main evaluation is complete. The report is a working draft uploaded on 21 November 2026 and still needs updating. The summarisation follow-up is described separately below.

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
| Report | [Working-draft PDF](docs/Multi_agent_Benchmark_Report_draft.pdf), including methods, figures, detailed results and scoring appendices |
| Main evaluation | [Summary](results/eval100/summary.json), [question/round outcomes](results/eval100/questions.json) and [run metadata](results/eval100/run_metadata.json) |
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
```

```python
!python -u -m debate.runner --manifest manifests/evaluation.json --repeats 3 --output /content/drive/MyDrive/sakana-results/eval100-reproduction --evaluation-ready
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

Revision rounds substantially increased token usage, which motivated a [summarisation pilot](results/README.md#summarisation-development-pilots) on five development questions and one repeat. All three variants answered 5/5 correctly. Summarising each initial answer once and reusing it reduced total token usage by 4.65%, but summary and revision time increased by 32.76%. All five initial group predictions were already correct, so this pilot does not establish whether summarisation preserves accuracy or helps correct mistakes. Results from the full 100-question follow-up have not been added yet.

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
| `docs/` | Draft report and reproducibility details |
| `prior_test_exposure.json` | Exposure inventory used by dataset selection |
| `requirements-colab.txt` | Pinned GPU runtime dependencies |

## AI assistance

ChatGPT, Codex and Claude were used to help with implementation, debugging, repository documentation, and drafting or editing parts of the report.
