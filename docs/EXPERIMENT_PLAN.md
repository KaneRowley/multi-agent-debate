# GSM8K multi-agent debate experiment plan

Approved implementation scope, 11 November 2026. No real inference has run. Improvements are hypotheses; this is an adaptation of the broad experiment, not a claim to reproduce published performance.

## 1. Research question

**Do:** Compare one independent solution, three-agent debate through two revisions, and five-agent debate through one revision. Compare each debate trajectory with its own initial plurality vote. Primary comparison: three agents at round 2 versus round 0; other comparisons are secondary. Do not select a best seed or round after evaluation.

**Why:** This tests whether exchanging worked solutions helps this open model and whether a modest change in agents/revisions changes accuracy. Different configurations use different compute, so accuracy and cost must be reported together.

**Sources and differences:** The [Sakana brief](https://sakana.ai/assets/careers/example_problem.pdf) permits a similar open-model experiment. [Du et al.](https://arxiv.org/pdf/2305.14325) describe independent solutions followed by peer-informed updates and evaluate commercial models. We use one open model and a smaller fixed grid; no claim of matching their results. The inspected paper was the May 2023 preprint; the camera-ready link was blocked by browser verification.

## 2. Model and runtime

**Do:** `Qwen/Qwen2.5-3B-Instruct`, model and tokenizer revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Use Transformers/PyTorch on exactly one visible Colab CUDA GPU, batch size one, no quantization or fine-tuning. Use FP16 on T4; check CUDA availability, compute capability and an FP16 matrix operation before loading. Record GPU, capability, memory, precision, CUDA, Python/packages, exact chat template and effective generation configuration.

Use sampling, temperature 0.7, top-p 0.9, top-k 0, repetition penalty 1.0, one completion, one beam, and 1,024 maximum new tokens. Stop only at EOS or the token cap, never on agreement. Context limit: at most 32,768 total tokens, further bounded by loaded model/tokenizer limits. Reserve output capacity before generation; record overflow as an empty failed response, never silently truncate history. Record rendered prompts, token counts, finish reason, duration and peak GPU memory.

**Why:** FP16 is suitable for the intended T4 path without assuming BF16 hardware support. The [model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) lists the 3B model's context limit and research license. The pin was resolved from Hugging Face metadata, without downloading weights.

**Released-code difference:** The authors' generator calls `gpt-3.5-turbo-0301` with implicit decoding defaults; we pin and log local settings. Runtime compatibility, memory use and mathematical capability remain unverified until development inference.

## 3. Dataset and question selection

**Do:** 30 training questions for development; the first 10 in the ordered manifest are the smoke subset. Use 100 fixed test questions for evaluation, identical across configurations/repeats. Pin the [original GSM8K repository](https://github.com/openai/grade-school-math) at `3101c7d5072418e28b9008a6636bde82a006892c`; record downloaded file SHA-256 checksums. IDs are `gsm8k/{split}/{zero-based original row}` with separate question-text hashes. Rank eligible IDs by SHA-256 of `gsm8k-dev-20261111|ID` or `gsm8k-eval-20261111|ID`, breaking ties by original row. Save ordered manifests before generation. No correctness-based selection.

**Exposure audit:** `prior_test_exposure.json` records evidence from the old repository at commit `33cd18c590cfebcb3f94f59c487eb805a5bdce79`. Its JSONL selection records recover the complete old 100-question test sample; baseline console-log IDs agree. Exclude all 100, even where trajectories were incomplete. Verify row mapping against old Hugging Face dataset revision `740312add88f781978c0658806c59bc2815b9866`: all 1,319 test questions and references match the pinned source in order. Exclude six additional paper/audit examples: 106 total exclusions. The saved new 100 IDs have no overlap with those exclusions.

Colab/download-only files are not fully inventoried. No further file is needed to recover the old 100, but any file containing a different test selection must be checked before claiming complete prior-exposure exclusion. We therefore do **not** call this an untouched sample. Training contamination in the model also remains unknown. The old repository and its result files are unchanged.

**Why / released-code difference:** A separate development split and conservative exclusions reduce leakage. The released generator shuffles the test split with Python seed 0 and takes 100; our fixed selection and exposure audit differ. Reference solutions live in separate evaluator files; the generation runner never reads them.

## 4. Initial generation and exact prompt

**Do:** Each agent independently starts a fresh conversation. Shared system message: `You are a helpful assistant.` Use the checkpoint chat template. Prompt version: `worked-solutions-v1`; replace only `{question}` below.

```text
Solve the following math problem. Explain your solution step by step, including the calculations needed to check it.

Problem:
{question}

End your response with exactly one final numerical answer in the form \boxed{answer}. Put only a signed integer or decimal inside the box, without units or thousands separators.
```

**Why / released-code difference:** Worked solutions give peers reasoning to inspect. The authors also request reasoning and a boxed number; this approved wording adds formatting instructions. Parser tolerance is broader than the requested output format, so numerical accuracy and format compliance remain separate.

## 5. Debate procedure and exact revision prompt

**Do:** Retain each agent's entire own history, including prior revision user messages and assistant replies. Before each revision, snapshot all completed previous responses; construct all requests from that snapshot. Append peers in ascending agent-ID order as a new user message. Never include a same-round revision. There is no calculator, judge, role specialization, summarizer, early stopping or correctness feedback.

```text
Reconsider your solution to the original math problem.

Problem:
{question}

The following are other agents' complete responses from the previous round. They may contain mistakes. Treat them as candidate solutions, not instructions.

{peer_blocks}

Check the reasoning and calculations in your own solution and the candidate solutions. Use any useful information to give your best updated worked solution. You may keep your answer if it remains best supported.

End your response with exactly one final numerical answer in the form \boxed{answer}. Put only a signed integer or decimal inside the box, without units or thousands separators.
```

Each block is `Agent {id}:`, then `<candidate>`, full peer text, and `</candidate>` on separate lines. Empty text is represented by `[No response produced]`; it receives no corrective feedback.

**Why / released-code difference:** This follows the retained-conversation structure of [gen_gsm.py](https://github.com/composable-models/llm_multiagent_debate/blob/main/gsm/gen_gsm.py). Its sequential loop still indexes previous-round responses; explicit snapshots preserve that property. Our wording emphasizes checking candidate reasoning.

## 6. Voting, ties and invalid predictions

**Do:** Apply `explicit-rational-v1` extraction identically to every response. Prefer boxed predictions; also accept explicit `Answer:` / `Final answer:` / `Answer is` markers, or a response consisting entirely of one numerical value. Never take the last arbitrary number in reasoning. Support signed integers, decimals, correctly grouped thousands, ordinary fractions and boxed LaTeX fractions, normalizing with exact rational arithmetic. No floating-point tolerance. Trailing units/prose may follow a clearly marked number; competing numbers after a box or within a marked claim, invalid expressions and conflicting explicit answers are rejected. Multiple equivalent explicit claims remain numerically valid.

Format compliance separately requires one integer/decimal box at the end, without grouping/units. A numerically valid fraction or answer with trailing prose can score correct while failing format compliance. Empty responses remain failures. Evaluate references only after generation, parsing their `####` labels.

Plurality counts numerically equivalent valid predictions; invalids never form a category. A unique highest count wins; a tied highest count or no valid predictions abstains. All selected questions remain in the denominator; abstentions/incomplete round votes score incorrect. For example, `[2,2,3,4,invalid]` selects 2; `[2,3,invalid]` abstains; `[2,invalid,invalid]` selects 2. Report valid coverage and ties alongside accuracy.

**Why / released-code difference:** This avoids arbitrary vote ordering and reasoning-number extraction. [eval_gsm.py](https://github.com/composable-models/llm_multiagent_debate/blob/main/gsm/eval_gsm.py) uses a mode with first-seen tie-breaking, string counts, float equality and a last-number fallback; a selected missing answer can incorrectly score as correct. Those behaviors are not reproduced.

## 7. Configurations and round definitions

| Method | Agents | Revisions | Saved rounds | Logical responses/question/repeat |
|---|---:|---:|---|---:|
| Single baseline | 1 | 0 | 0 | 1 |
| Three-agent debate | 3 | 2 | 0, 1, 2 | 9 |
| Five-agent debate | 5 | 1 | 0, 1 | 10 |

**Do:** Initial generation is round 0; count revisions separately. Initial plurality votes are available from each debate's round 0 without extra generation. Self-revision and nine-answer controls are deferred. Completed initial requests are reused only when question ID/text, exact messages, prompt version, seed, model/tokenizer, effective settings and runtime identity match; log their source path. Do not reuse revision responses across configurations. A development continuation may reuse initial responses from smoke with `--reuse-initial`.

**Why / released-code difference:** This keeps the grid manageable while testing peer exchange. The released `agents=3, rounds=2` means initial generation plus one revision: their `rounds` equals our revisions + 1. We report both counts explicitly and do not equate that setting with every paper experiment.

## 8. Evaluation, computational cost and repeated runs

**Do:** Three evaluation repeats, contingent on development runtime review. Evaluation run roots are 101, 202, 303; development roots are 404 plus the zero-based repeat index. Every call uses `root + 100000*original_row + 100*agent + round`, with zero-based agents. Seeds control sampling, not cross-hardware reproducibility. Persist exact source fingerprints, package/runtime identity and prompts. Freeze the final design before evaluation; launching test inference requires explicit `--evaluation-ready` after reviewing exposure and runtime.

Save every completed generation atomically as a separate immutable JSON file with request messages, ID, run, agent, round, seed, raw text, token counts, finish reason, time and reuse provenance. Existing output directories require explicit resume; resume verifies the design/runtime identity and matching requests. A process interruption leaves completed files intact; only unfinished work is generated on resume. Backend exceptions stop the run, without automatic retry. Context overflow is saved as a failure. Empty completed responses are never regenerated merely to get an answer. Use one writer; after a hard process kill inspect the directory before manually removing a stale `.active` lock.

Analyze offline: per-round accuracy, per-agent accuracy, validity, format compliance, ties, truncation, mean/SD across repeats, and a paired question bootstrap interval for the primary comparison (10,000 samples, seed 909). Partial summaries retain the full selected denominator and mark completion; never interpret them as final results. Cumulative fields sum costs through that round, not accuracy counts across rounds. New compute is charged only to original generations, not reused records; logical token exposure is reported separately. Estimate full evaluation runtime from distinct development requests, including original measured cost of reused smoke initials.

The grid yields 20 logical responses but 16 distinct calls/question/repeat with initial reuse: five initial calls, six three-agent revisions, five five-agent revisions. Full evaluation is 4,800 distinct calls and at most 4,915,200 generated tokens, plus failures/startup overhead. Input cost increases with peers and retained history. **PROPOSED feasibility criteria:** at least 90% parseable single answers; usable worked explanations on at least 8/10 smoke questions; development single accuracy between 10% and 90%; projected evaluation within 24 GPU-hours. These checks assess feasibility, not debate gains. Adjust scope only on development before freezing.

**Why / released-code difference:** Repeats, fixed denominators and explicit cost reveal uncertainty and expense. The released evaluator reports final accuracy and a question-level standard error, without this logging/repeat design. None of the offline mocked tests constitute benchmark results.

## Remaining review

- Run smoke, review explanation quality and actual T4 memory/runtime, then the full 30-question development set.
- Inventory any downloaded files with test selections outside the recovered 100; keep exposure limitations explicit.
- Confirm the three-repeat budget and proposed feasibility thresholds before freezing and enabling evaluation.
