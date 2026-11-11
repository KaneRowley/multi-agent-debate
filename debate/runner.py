"""Retained histories, synchronous snapshots and immutable per-generation records."""
import argparse
import copy
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from . import prompts
from .config import CONFIGURATIONS
from .dataset import validate_manifest
from .storage import digest, read, save_new
from .continuation import SEED_VERSION, load_source, request_for


def run(manifest, generator, output, *, repeats=1, limit=None, resume=False,
        reuse_initial=None, continue_from=None, prepare_only=False):
    validate_manifest(manifest)
    if repeats not in (1, 2, 3):
        raise ValueError("Use 1–3 repeats")
    if limit is not None and (manifest["split"] != "train" or limit not in (10, 30)):
        raise ValueError("Only development supports limits: 10 or 30")
    if limit and len(manifest["questions"]) < limit:
        raise ValueError("Not enough development questions")
    questions = manifest["questions"][:limit] if limit else manifest["questions"]
    if reuse_initial and continue_from:
        raise ValueError("Use --continue-from for full trajectories, or --reuse-initial, not both")
    if prepare_only and not continue_from:
        raise ValueError("--prepare-only requires --continue-from")
    if reuse_initial and not (Path(reuse_initial) / "run.json").is_file():
        raise FileNotFoundError("Initial-reuse source must be an existing run directory")
    output = Path(output)
    if continue_from and (output.resolve().is_relative_to(Path(continue_from).resolve())
                          or Path(continue_from).resolve().is_relative_to(output.resolve())):
        raise ValueError("Continuation output must differ from and remain separate from the preserved source directory")
    source_records, source_hashes, continuation = {}, {}, None
    if continue_from:
        source_records, source_hashes, continuation = load_source(
            continue_from, manifest, [q["id"] for q in questions], repeats, generator.identity)
    metadata = dict(schema_version=1, manifest_sha256=digest(manifest), manifest=manifest,
                    selected_ids=[q["id"] for q in questions], repeats=repeats,
                    configurations=CONFIGURATIONS, generator=generator.identity,
                    reuse_initial=str(Path(reuse_initial).resolve()) if reuse_initial else None,
                    continuation=continuation,
                    prompt_version=prompts.PROMPT_VERSION, initial_prompt=prompts.INITIAL,
                    revision_prompt=prompts.REVISION, system=prompts.SYSTEM,
                    source_sha256=digest({p.name: p.read_text(encoding="utf-8")
                                          for p in Path(__file__).parent.glob("*.py")}),
                    seed_version=SEED_VERSION)
    # JSON canonicalization also converts tuples to lists for resume comparisons.
    if output.exists() and not resume:
        raise FileExistsError("Output exists; use explicit --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".active"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        if (output / "run.json").exists():
            saved = read(output / "run.json")
            # Source snapshots remain audit evidence. Explicit generation contracts
            # and every replayed request govern compatibility, not parser/analysis
            # implementation changes. Older metadata has no continuation field.
            comparable = lambda value: {k: v for k, v in value.items()
                                        if k != "source_sha256" and not (k == "continuation" and v is None)}
            if digest(comparable(saved)) != digest(comparable(metadata)):
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
            for question in questions:
                for method, (agents, revisions) in CONFIGURATIONS.items():
                    histories = [prompts.initial(question["question"]) for _ in range(agents)]
                    previous = [""] * agents
                    for round_index in range(revisions + 1):
                        if round_index and any(text is None for text in previous):
                            # Prepare-only cannot construct revisions of missing responses.
                            break
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
                            request = request_for(question, messages, generator.identity,
                                                  run_index, agent, round_index, manifest["split"])
                            seed = request["seed"]
                            key = digest(request)
                            filename = f"run{run_index}-{method}-q{question['index']}-r{round_index}-a{agent}.json"
                            path = output / "generations" / filename
                            source_slot = (run_index, method, question["id"], round_index, agent)
                            if path.exists():
                                record = read(path)
                                if record["request_key"] != key or record["request"] != request:
                                    raise ValueError("Existing generation does not match request")
                                if source_slot in source_records:
                                    source_path, origin = source_records[source_slot]
                                    if (record.get("source_record_sha256") != source_hashes[source_path.name]
                                            or record.get("source_run_sha256") != continuation["run_sha256"]
                                            or record.get("source_record") != source_path.name
                                            or record.get("reuse_kind") != "continuation"
                                            or any(record.get(k) != origin[k] for k in origin["result_fields"])):
                                        raise ValueError("Imported generation result/provenance mismatch")
                            else:
                                reused_from = None
                                provenance = {}
                                if source_slot in source_records:
                                    source_path, origin = source_records[source_slot]
                                    if origin["request"] != request or origin["request_key"] != key:
                                        raise ValueError("Completed source request changed; refusing regeneration")
                                    reused_from = str(source_path.resolve())
                                    provenance = dict(reuse_kind="continuation", source_record=source_path.name,
                                        source_run_sha256=continuation["run_sha256"],
                                        source_record_sha256=source_hashes[source_path.name],
                                        source_reused_from=origin["reused_from"])
                                    result = {k: origin[k] for k in origin["result_fields"]}
                                elif round_index == 0 and key in initial_cache:
                                    reused_from, origin = initial_cache[key]
                                    result = {k: origin[k] for k in origin["result_fields"]}
                                else:
                                    if prepare_only:
                                        current.append(None)
                                        continue
                                    # Exceptions stop the run; completed records remain durable.
                                    # An uncompleted request may be attempted on explicit resume.
                                    result = generator.generate(copy.deepcopy(messages), seed)
                                record = dict(question_id=question["id"], run=run_index, method=method,
                                              recorded_utc=datetime.now(timezone.utc).isoformat(),
                                              agent=agent, round=round_index, seed=seed,
                                              request_key=key, request=request, reused_from=reused_from,
                                              result_fields=list(result), **provenance, **result)
                                save_new(path, record)
                            if round_index == 0 and record["finish_reason"] not in ("error", "context_overflow"):
                                initial_cache.setdefault(key, (str(path.resolve()), record))
                            current.append(record["raw_response"])
                            histories[agent] = messages + [dict(role="assistant", content=record["raw_response"])]
                        previous = current
    finally:
        lock.unlink()
    records = list((output / "generations").glob("*.json"))
    expected = repeats * len(questions) * sum(a * (r + 1) for a, r in CONFIGURATIONS.values())
    return dict(selected_questions=len(questions), expected_responses=expected,
                saved_responses=len(records), missing_responses=expected - len(records),
                source_responses=len(source_records), prepare_only=prepare_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-initial", help="Read matching initial responses from one prior run directory")
    parser.add_argument("--continue-from", help="Replay every completed response/revision from a matching completed prefix run")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and import saved work without loading a model")
    parser.add_argument("--evaluation-ready", action="store_true",
                        help="Attest design freeze and review of development runtime and exposure audit")
    args = parser.parse_args()
    manifest = read(args.manifest)
    validate_manifest(manifest)
    if manifest["split"] == "test" and not args.evaluation_ready:
        parser.error("Evaluation requires design/runtime/exposure review and --evaluation-ready")
    if Path(args.output).exists() and not args.resume:
        parser.error("Output already exists; use --resume or a new directory")
    if args.prepare_only:
        if not args.continue_from:
            parser.error("--prepare-only requires --continue-from")
        generator = SimpleNamespace(identity=read(Path(args.continue_from) / "run.json")["generator"])
    else:
        from .generation import Generator
        generator = Generator()
    result = run(manifest, generator, args.output, repeats=args.repeats, limit=args.limit,
                 resume=args.resume, reuse_initial=args.reuse_initial,
                 continue_from=args.continue_from, prepare_only=args.prepare_only)
    print(result)


if __name__ == "__main__":
    main()
