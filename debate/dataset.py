"""Deterministic selection; labels are stored separately from generation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from .config import DATASET_REVISION
from .storage import digest, read, save_new


def source(split):
    url = (f"https://raw.githubusercontent.com/openai/grade-school-math/{DATASET_REVISION}"
           f"/grade_school_math/data/{split}.jsonl")
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    return [json.loads(line) for line in raw.splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


def select(rows, split, count, exclusions=()):
    candidates = []
    for i, row in enumerate(rows):
        qid = f"gsm8k/{split}/{i}"
        if qid not in exclusions:
            candidates.append(dict(id=qid, index=i, question=row["question"],
                                   question_sha256=hashlib.sha256(row["question"].encode()).hexdigest()))
    salt = "gsm8k-dev-20261111" if split == "train" else "gsm8k-eval-20261111"
    candidates.sort(key=lambda q: (hashlib.sha256((salt + "|" + q["id"]).encode()).hexdigest(), q["index"]))
    if len(candidates) < count:
        raise ValueError("Not enough eligible questions")
    return candidates[:count]


def validate_manifest(manifest):
    if manifest["split"] not in ("train", "test"):
        raise ValueError("Unknown split")
    questions = manifest["questions"]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Empty or duplicate question manifest")
    for q in questions:
        expected_id = f"gsm8k/{manifest['split']}/{q['index']}"
        if q["id"] != expected_id or q["question_sha256"] != hashlib.sha256(q["question"].encode()).hexdigest():
            raise ValueError("Question identity mismatch")
        if set(q) != {"id", "index", "question", "question_sha256"}:
            raise ValueError("Generation manifest contains unexpected fields (possibly labels)")
    if manifest["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Unexpected dataset revision")


def prepare(output, exposure):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new manifest directory; existing selections are immutable")
    test, test_sha = source("test")
    train, train_sha = source("train")
    excluded = set(exposure["excluded_ids"])
    if not excluded.issubset({f"gsm8k/test/{i}" for i in range(len(test))}):
        raise ValueError("Unknown exclusion ID")
    for pair in exposure.get("verified_question_text", []):
        i = int(pair["id"].rsplit("/", 1)[1])
        if test[i]["question"] != pair["question"]:
            raise ValueError("Old/new dataset row mapping differs; audit before evaluation")
    plans = [("development", "train", train, train_sha, 30),
             ("evaluation", "test", test, test_sha, 100)]
    output.mkdir(parents=True)
    for name, split, rows, checksum, count in plans:
        selected = select(rows, split, count, excluded if split == "test" else ())
        manifest = dict(schema_version=1, split=split, dataset_revision=DATASET_REVISION,
                        source_sha256=checksum, selection_version="sha256-rank-v1",
                        exposure=exposure if split == "test" else None, questions=selected)
        validate_manifest(manifest)
        save_new(output / f"{name}.json", manifest)
        save_new(output / f"{name}_references.json",
                 {q["id"]: rows[q["index"]]["answer"] for q in selected})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--exposure", required=True)
    args = p.parse_args()
    prepare(args.output, read(args.exposure))


if __name__ == "__main__":
    main()
