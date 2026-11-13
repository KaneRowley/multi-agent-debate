# Multi-agent debate on GSM8K

Does sharing worked solutions improve a language model's mathematical accuracy? This project compares a single agent, independent plurality voting, and multi-agent debate using **Qwen2.5-3B-Instruct** on **100 fixed GSM8K test questions**, with **three generation-seed repeats**.

The evaluation is complete. Initial voting improves on the single-agent baseline; debate adds a smaller average gain. The primary debate comparison remains uncertain on this sample.

## Results

Mean accuracy across three repeats, using parser `explicit-rational-v3`:

| Configuration | Initial voting | One revision | Two revisions |
|---|---:|---:|---:|
| Single agent | 87.00% | — | — |
| Three agents | 89.67% | 91.33% | 91.67% |
| Five agents | 92.67% | 93.67% | — |

**Round 0 is independent generation.** A revision lets each agent review the other agents' previous worked solutions while retaining its own conversation. The table reports every evaluated configuration; a dash means that configuration was not run.

The primary comparison, three agents after two revisions versus their initial vote, improves by **2.00 percentage points**. Its paired question-bootstrap 95% interval is approximately **−0.33 to +4.67 points**, so the additional benefit of debate is not established conclusively. The three repeats use the same 100 questions, not 300 independent questions.

The workload contains 6,000 saved response records: 4,800 actual model calls and 1,200 exact-request reuses. Recorded generation time totals about 19.53 T4 GPU-hours, excluding startup, file I/O and interruptions. This is the combined experiment cost, not the standalone cost of any one configuration.

See the [final experiment record](docs/FINAL_EXPERIMENT.md) for model pins, archive checksums and validation. Raw generations and analyses are preserved in the separate `eval100-v3-review.zip` backup; that archive is not committed to this repository. The report and figures are still being prepared.

## Method

- **Model:** `Qwen/Qwen2.5-3B-Instruct`, pinned model/tokenizer revision; Transformers/PyTorch, FP16 on a Tesla T4.
- **Generation:** temperature 0.7, top-p 0.9, top-k 0, at most 1,024 new tokens.
- **Development:** 30 fixed GSM8K training questions. Evaluation uses 100 fixed test questions and seed roots 101, 202 and 303.
- **Debate:** independent initial answers, then synchronous previous-round peer sharing. No judge, calculator, role specialization or agreement-based early stopping.
- **Scoring:** explicit numerical extraction, exact rational equivalence, and plurality voting. Invalid answers do not vote; tied votes abstain and score incorrect. Format compliance is measured separately.
- **Selection:** evaluation IDs exclude 106 documented prior-exposure IDs. Additional download-only exposure and model pretraining contamination remain limitations; the sample is not claimed to be entirely untouched.

This adapts [Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325). Model, prompts, selection, scale and scoring differ from the published experiment. The [original design plan](docs/EXPERIMENT_PLAN.md) and [documentation index](docs/README.md) record these choices and the development history.

## Repository layout

| Location | Purpose |
|---|---|
| `debate/` | Dataset selection, prompts, generation, runner, scoring and analysis |
| `manifests/` | Fixed questions and separate evaluator references |
| `tests/` | Offline software tests and response fixtures |
| `docs/` | Design plan and final experiment record |
| `docs/archive/` | Historical development instructions and audits |
| `prior_test_exposure.json` | Exposure inventory used by dataset selection |
| `requirements-colab.txt` | Pinned GPU runtime dependencies |

## Offline checks

From the repository root, using Python 3.11 or later:

```bash
python -m unittest discover -s tests -v
```

These standard-library tests do not download a model or run inference. The preserved implementation passed 28 offline tests, and an offline replay reproduced all 1,800 saved v3 question/round outcomes.

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

For a reproduction of the frozen evaluation, mount Drive so completed records survive runtime deletion:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!python -u -m debate.runner --manifest manifests/evaluation.json --repeats 3 --output /content/drive/MyDrive/sakana-results/eval100 --evaluation-ready
```

The first generation run downloads model weights. `--evaluation-ready` explicitly acknowledges that the design is frozen; the command above reproduces the existing experiment rather than requesting a new design search.

To continue a stopped run, repeat its exact command with `--resume` appended. Restore the same code, GPU/runtime settings and Drive mount first. Resume checks compatibility and skips completed records. Run only one writer per output directory. If a hard interruption leaves `.active`, confirm the old process has ended before removing that lock. Never delete completed response files to obtain different answers.

Local `/content` storage is temporary. Keep both the final code and results ZIP backups outside it. Historical smoke-to-development continuation instructions are retained under [the archive](docs/archive/README.md); their v2 instructions describe an earlier stage, not the final scoring setup.
