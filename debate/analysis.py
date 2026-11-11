"""Offline scoring and cost summaries; this module alone reads references."""
import argparse
import random
import statistics
from pathlib import Path

from .config import CONFIGURATIONS
from .scoring import PARSER_VERSION, extract, plurality, reference
from .storage import digest, read, save_new


def costs(records):
    fresh = [r for r in records if r["reused_from"] is None]
    return dict(logical_responses=len(records), reused_responses=len(records) - len(fresh),
                model_calls=sum(r["model_call"] for r in fresh),
                input_tokens=sum(r["input_tokens"] for r in fresh),
                output_tokens=sum(r["output_tokens"] for r in fresh),
                generation_seconds=sum(r["seconds"] for r in fresh),
                logical_input_tokens=sum(r["input_tokens"] for r in records),
                logical_output_tokens=sum(r["output_tokens"] for r in records))


def analyze(directory, references):
    directory = Path(directory)
    metadata = read(directory / "run.json")
    rows = [read(p) for p in sorted((directory / "generations").glob("*.json"))]
    selected = set(metadata["selected_ids"])
    if not selected.issubset(references):
        raise ValueError("Missing reference labels")
    index = {}
    for row in rows:
        key = (row["run"], row["method"], row["question_id"], row["round"], row["agent"])
        if key in index or row["question_id"] not in selected:
            raise ValueError("Duplicate or unexpected generation")
        index[key] = row
    details, summaries = [], []
    for run_index in range(metadata["repeats"]):
        for method, (agents, revisions) in CONFIGURATIONS.items():
            for round_index in range(revisions + 1):
                stage = []
                for qid in metadata["selected_ids"]:
                    records = [index.get((run_index, method, qid, round_index, a)) for a in range(agents)]
                    predictions = [extract(r["raw_response"]) if r else extract("") for r in records]
                    vote, status = plurality([p.value for p in predictions])
                    # Partial rounds cannot count as completed votes; keep denominator fixed.
                    complete = all(r is not None for r in records)
                    detail = dict(run=run_index, method=method, question_id=qid, round=round_index,
                                  complete=complete, prediction=str(vote) if complete and vote is not None else None,
                                  vote_status=status if complete else "incomplete",
                                  correct=bool(complete and vote is not None and vote == reference(references[qid])),
                                  valid_predictions=sum(p.value is not None for p in predictions),
                                  format_compliant=sum(p.format_compliant for p in predictions),
                                  extraction_statuses=[p.status for p in predictions],
                                  agent_correct=sum(p.value is not None and p.value == reference(references[qid])
                                                    for p in predictions))
                    stage.append(detail)
                exact = [r for r in rows if r["run"] == run_index and r["method"] == method and r["round"] == round_index]
                cumulative = [r for r in rows if r["run"] == run_index and r["method"] == method and r["round"] <= round_index]
                summaries.append(dict(run=run_index, method=method, round=round_index,
                    questions=len(selected), completed_questions=sum(x["complete"] for x in stage),
                    accuracy=sum(x["correct"] for x in stage) / len(selected),
                    agent_accuracy=sum(x["agent_correct"] for x in stage) / (len(selected) * agents),
                    format_compliance=sum(x["format_compliant"] for x in stage) / (len(selected) * agents),
                    invalid_rate=1 - sum(x["valid_predictions"] for x in stage) / (len(selected) * agents),
                    tie_rate=sum(x["vote_status"] == "tie" for x in stage) / len(selected),
                    truncated_responses=sum(r["finish_reason"] == "length" for r in exact),
                    round_cost=costs(exact), cumulative_cost=costs(cumulative)))
                details.extend(stage)
    aggregate = []
    for method, (_, revisions) in CONFIGURATIONS.items():
        for r in range(revisions + 1):
            group = [s for s in summaries if s["method"] == method and s["round"] == r]
            values = [s["accuracy"] for s in group]
            aggregate.append(dict(method=method, round=r, mean_accuracy=statistics.mean(values),
                                  sd_accuracy=statistics.stdev(values) if len(values) > 1 else None))
    all_complete = all(s["completed_questions"] == len(selected) for s in summaries)
    total_cost = costs(rows)
    unique_workload = list({r["request_key"]: r for r in rows}.values())
    workload_seconds = sum(r["seconds"] for r in unique_workload)
    projected = None
    if metadata["manifest"]["split"] == "train" and all_complete:
        projected = workload_seconds / (len(selected) * metadata["repeats"]) * 100 * 3 / 3600
    primary = None
    if all_complete:
        lookup = {(d["run"], d["method"], d["round"], d["question_id"]): d for d in details}
        differences = [statistics.mean(
            int(lookup[(run, "debate3", 2, qid)]["correct"]) - int(lookup[(run, "debate3", 0, qid)]["correct"])
            for run in range(metadata["repeats"])) for qid in metadata["selected_ids"]]
        rng = random.Random(909)
        bootstrap = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
        primary = dict(comparison="debate3 r2 minus r0", difference=statistics.mean(differences),
                       question_bootstrap_95=[bootstrap[250], bootstrap[9749]], seed=909,
                       caveat="Question sampling uncertainty; repeats are retained within each question.")
    return dict(parser_version=PARSER_VERSION, run_sha256=digest(metadata), complete=all_complete,
                per_round=summaries, across_runs=aggregate, actual_total_cost=total_cost,
                distinct_workload_seconds_including_reused=workload_seconds,
                projected_test_3_repeats_gpu_hours=projected,
                runtime_estimate_caveat="Generation only; longer test prompts, startup and I/O may add cost.",
                primary_comparison=primary), details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Analysis output exists; choose a new snapshot directory")
    summary, details = analyze(args.run, read(args.references))
    save_new(output / "summary.json", summary)
    save_new(output / "questions.json", details)
    print(f"Saved offline analysis to {output}; complete={summary['complete']}")


if __name__ == "__main__":
    main()
