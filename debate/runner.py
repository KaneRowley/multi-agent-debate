"""Retained histories, synchronous snapshots and immutable per-generation records."""
import argparse
import copy
import os
from datetime import datetime, timezone
from pathlib import Path

from . import prompts
from .config import CONFIGURATIONS, RUN_SEEDS, DEV_SEED
from .dataset import validate_manifest
from .storage import digest, read, save_new


def run(manifest, generator, output, *, repeats=1, limit=None, resume=False, reuse_initial=None):
    validate_manifest(manifest)
    if repeats not in (1, 2, 3):
        raise ValueError("Use 1–3 repeats")
    if limit is not None and (manifest["split"] != "train" or limit not in (10, 30)):
        raise ValueError("Only development supports limits: 10 or 30")
    if limit and len(manifest["questions"]) < limit:
        raise ValueError("Not enough development questions")
    questions = manifest["questions"][:limit] if limit else manifest["questions"]
    if reuse_initial and not (Path(reuse_initial) / "run.json").is_file():
        raise FileNotFoundError("Initial-reuse source must be an existing run directory")
    output = Path(output)
    metadata = dict(schema_version=1, manifest_sha256=digest(manifest), manifest=manifest,
                    selected_ids=[q["id"] for q in questions], repeats=repeats,
                    configurations=CONFIGURATIONS, generator=generator.identity,
                    reuse_initial=str(Path(reuse_initial).resolve()) if reuse_initial else None,
                    prompt_version=prompts.PROMPT_VERSION, initial_prompt=prompts.INITIAL,
                    revision_prompt=prompts.REVISION, system=prompts.SYSTEM,
                    source_sha256=digest({p.name: p.read_text(encoding="utf-8")
                                          for p in Path(__file__).parent.glob("*.py")}),
                    seed_version="root+100000*row+100*agent+round-v1")
    # JSON canonicalization also converts tuples to lists for resume comparisons.
    if output.exists() and not resume:
        raise FileExistsError("Output exists; use explicit --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".active"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        if (output / "run.json").exists():
            if digest(read(output / "run.json")) != digest(metadata):
                raise ValueError("Resume metadata mismatch; use a new output directory")
        else:
            save_new(output / "run.json", metadata)
        initial_cache = {}
        # Index existing initial records before doing any further generation.
        sources = [output] + ([Path(reuse_initial)] if reuse_initial else [])
        for path in sorted(p for source in sources for p in (source / "generations").glob("*.json")):
            record = read(path)
            if digest(record["request"]) != record["request_key"]:
                raise ValueError("Invalid cached request fingerprint")
            if record["round"] == 0 and record["finish_reason"] not in ("error", "context_overflow"):
                initial_cache.setdefault(record["request_key"], (str(path.resolve()), record))
        for run_index in range(repeats):
            root = RUN_SEEDS[run_index] if manifest["split"] == "test" else DEV_SEED + run_index
            for question in questions:
                for method, (agents, revisions) in CONFIGURATIONS.items():
                    histories = [prompts.initial(question["question"]) for _ in range(agents)]
                    previous = [""] * agents
                    for round_index in range(revisions + 1):
                        snapshot = tuple(previous)
                        requests = []
                        for agent in range(agents):
                            history = copy.deepcopy(histories[agent])
                            if round_index:
                                history.append(dict(role="user", content=prompts.revision(
                                    question["question"], snapshot, agent)))
                            requests.append(history)
                        current = []
                        for agent, messages in enumerate(requests):
                            seed = root + 100000 * question["index"] + 100 * agent + round_index
                            request = dict(question_id=question["id"], question=question["question"],
                                           messages=messages, seed=seed, generator=generator.identity,
                                           prompt_version=prompts.PROMPT_VERSION)
                            key = digest(request)
                            filename = f"run{run_index}-{method}-q{question['index']}-r{round_index}-a{agent}.json"
                            path = output / "generations" / filename
                            if path.exists():
                                record = read(path)
                                if record["request_key"] != key or record["request"] != request:
                                    raise ValueError("Existing generation does not match request")
                            else:
                                reused_from = None
                                if round_index == 0 and key in initial_cache:
                                    reused_from, origin = initial_cache[key]
                                    result = {k: origin[k] for k in origin["result_fields"]}
                                else:
                                    # Exceptions stop the run; completed records remain durable.
                                    # An uncompleted request may be attempted on explicit resume.
                                    result = generator.generate(copy.deepcopy(messages), seed)
                                record = dict(question_id=question["id"], run=run_index, method=method,
                                              recorded_utc=datetime.now(timezone.utc).isoformat(),
                                              agent=agent, round=round_index, seed=seed,
                                              request_key=key, request=request, reused_from=reused_from,
                                              result_fields=list(result), **result)
                                save_new(path, record)
                            if round_index == 0 and record["finish_reason"] not in ("error", "context_overflow"):
                                initial_cache.setdefault(key, (str(path.resolve()), record))
                            current.append(record["raw_response"])
                            histories[agent] = messages + [dict(role="assistant", content=record["raw_response"])]
                        previous = current
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-initial", help="Read matching initial responses from one prior run directory")
    parser.add_argument("--evaluation-ready", action="store_true",
                        help="Attest design freeze and review of development runtime and exposure audit")
    args = parser.parse_args()
    manifest = read(args.manifest)
    validate_manifest(manifest)
    if manifest["split"] == "test" and not args.evaluation_ready:
        parser.error("Evaluation requires design/runtime/exposure review and --evaluation-ready")
    if Path(args.output).exists() and not args.resume:
        parser.error("Output already exists; use --resume or a new directory")
    from .generation import Generator
    run(manifest, Generator(), args.output, repeats=args.repeats, limit=args.limit,
        resume=args.resume, reuse_initial=args.reuse_initial)


if __name__ == "__main__":
    main()
