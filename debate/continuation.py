"""Validate and replay a completed source trajectory without loading a model."""
import copy
import hashlib
from pathlib import Path

from . import prompts
from .config import CONFIGURATIONS, DEV_SEED, RUN_SEEDS
from .storage import digest, read

SEED_VERSION = "root+100000*row+100*agent+round-v1"


def request_for(question, messages, identity, run_index, agent, round_index, split):
    root = RUN_SEEDS[run_index] if split == "test" else DEV_SEED + run_index
    return dict(question_id=question["id"], question=question["question"], messages=messages,
                seed=root + 100000 * question["index"] + 100 * agent + round_index,
                generator=identity, prompt_version=prompts.PROMPT_VERSION)


def slot(record):
    return (record["run"], record["method"], record["question_id"], record["round"], record["agent"])


def load_source(directory, manifest, selected_ids, repeats, identity):
    """Fail before generation if any source record or contract cannot be replayed."""
    directory = Path(directory)
    metadata = read(directory / "run.json")
    source_ids = metadata["selected_ids"]
    checks = {
        "manifest": digest(metadata["manifest"]) == digest(manifest)
                    and metadata["manifest_sha256"] == digest(manifest),
        "selection prefix": bool(source_ids) and source_ids == selected_ids[:len(source_ids)],
        "repeats": metadata["repeats"] == repeats,
        "configurations": digest(metadata["configurations"]) == digest(CONFIGURATIONS),
        "model/runtime/settings": metadata["generator"] == identity,
        "system prompt": metadata["system"] == prompts.SYSTEM,
        "initial prompt": metadata["initial_prompt"] == prompts.INITIAL,
        "revision prompt": metadata["revision_prompt"] == prompts.REVISION,
        "prompt version": metadata["prompt_version"] == prompts.PROMPT_VERSION,
        "seed rule": metadata["seed_version"] == SEED_VERSION,
    }
    failed = [name for name, matches in checks.items() if not matches]
    if failed:
        raise ValueError("Continuation mismatch: " + ", ".join(failed))
    records, hashes = {}, {}
    for path in sorted((directory / "generations").glob("*.json")):
        record = read(path)
        key = slot(record)
        if key in records or digest(record["request"]) != record["request_key"]:
            raise ValueError("Duplicate or invalid source request fingerprint")
        records[key] = (path, record)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = set()
    questions = {q["id"]: q for q in manifest["questions"]}
    for run_index in range(repeats):
        for qid in source_ids:
            question = questions[qid]
            for method, (agents, revisions) in CONFIGURATIONS.items():
                histories = [prompts.initial(question["question"]) for _ in range(agents)]
                previous = [""] * agents
                for round_index in range(revisions + 1):
                    snapshot = tuple(previous)
                    current = []
                    for agent in range(agents):
                        key = (run_index, method, qid, round_index, agent)
                        expected.add(key)
                        if key not in records:
                            raise ValueError("Continuation source is incomplete: " + str(key))
                        path, record = records[key]
                        messages = copy.deepcopy(histories[agent])
                        if round_index:
                            messages.append(dict(role="user", content=prompts.revision(
                                question["question"], snapshot, agent)))
                        request = request_for(question, messages, identity, run_index, agent, round_index, manifest["split"])
                        if record["request"] != request or record["request_key"] != digest(request) or record["seed"] != request["seed"]:
                            raise ValueError("Continuation request/history mismatch: " + path.name)
                        required = {"raw_response", "input_tokens", "output_tokens", "finish_reason", "seconds", "model_call"}
                        if not required.issubset(record["result_fields"]) or not isinstance(record["raw_response"], str):
                            raise ValueError("Incomplete saved generation result: " + path.name)
                        # A completed empty/length/overflow response is replayed too.
                        # Parsing success and correctness never affect eligibility.
                        current.append(record["raw_response"])
                        histories[agent] = messages + [dict(role="assistant", content=record["raw_response"])]
                    previous = current
    if set(records) != expected:
        raise ValueError("Unexpected source question, agent, method, round or run")
    provenance = dict(run_sha256=digest(metadata), records_sha256=digest(hashes),
                      selected_ids=source_ids, saved_responses=len(records))
    return records, hashes, provenance
