# evals/log_langsmith.py
"""
log_langsmith.py -- push eval snapshots to LangSmith as comparable experiments.

Every snapshot written by run_suite.py (baselines/*.json) becomes ONE LangSmith
experiment:

  - all experiments hang off one shared dataset ("rag-eval-statquest-suite",
    a single example that stands for "one full suite run"), so LangSmith's
    "Compare experiments" view puts them side by side;
  - every metric in the snapshot becomes a feedback score on that experiment
    (booleans -> 1/0), so all 21 metrics -- including latency, cost and the SLO
    flags -- are charted and comparable;
  - the run metadata (label, prompt hash, top_k, fetch_k, created_at) is attached
    to the experiment, so runs can be filtered by config;
  - candidates are also compared against baselines/baseline.json with the same
    rules as compare.py, and the verdict (PASS / REVIEW / FAIL) plus each
    metric's status (improved / flat / regressed / blocked) is logged with it.

Uploading is idempotent: an experiment is named after the snapshot file and its
created_at time, and one that already exists is skipped (use --force to replace
it).

Needs LANGSMITH_API_KEY in .env (LANGSMITH_ENDPOINT too, for a non-US region).

    python -m evals.log_langsmith                        # every baselines/*.json
    python -m evals.log_langsmith baselines/candidate.json
    python -m evals.log_langsmith --force                # re-upload existing ones
    python -m evals.run_suite --label "..." --langsmith  # run suite, then log it
"""

import argparse
import glob
import math
import os
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from evals.compare import compare, load

load_dotenv()

DATASET_NAME = "rag-eval-statquest-suite"
BASELINE_PATH = "baselines/baseline.json"
DEFAULT_GLOB = "baselines/*.json"

# How the verdict is stored as a score, so it can be charted like a metric.
VERDICT_SCORE = {"PASS": 1.0, "REVIEW": 0.5, "FAIL": 0.0}


# ============================================================
# 1. HELPERS
# ============================================================
def _score(value):
    """Metric value -> numeric feedback score, or None if it can't be one."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    return None


def experiment_name(path, meta):
    """Stable, unique name: '<file stem> @ <created_at>'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem} @ {meta.get('created_at', 'unknown')}"


def get_or_create_dataset(client):
    """The shared dataset every experiment points at (one 'suite run' example)."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="One example = one full run of the rag-eval-statquest "
                        "suite. Each experiment is one snapshot from baselines/.",
        )
    examples = list(client.list_examples(dataset_id=dataset.id, limit=1))
    if examples:
        example = examples[0]
    else:
        example = client.create_example(
            dataset_id=dataset.id,
            inputs={"suite": "rag-eval-statquest full eval suite"},
        )
    return dataset, example


def verdict_for(path, snapshot, baseline):
    """compare.py's verdict + per-metric statuses against the baseline, or None."""
    if baseline is None or os.path.abspath(path) == os.path.abspath(BASELINE_PATH):
        return None, {}
    verdict, rows = compare(baseline, snapshot)
    return verdict, {r["id"]: r["status"] for r in rows}


# ============================================================
# 2. LOG ONE SNAPSHOT
# ============================================================
def log_snapshot(client, path, dataset, example, baseline, force=False):
    snapshot = load(path)
    meta = snapshot.get("metadata", {})
    metrics = snapshot.get("metrics", {})
    name = experiment_name(path, meta)

    if client.has_project(name):
        if not force:
            print(f"  skip  {name}  (already logged; --force to replace)")
            return None
        client.delete_project(project_name=name)

    verdict, statuses = verdict_for(path, snapshot, baseline)
    is_baseline = os.path.abspath(path) == os.path.abspath(BASELINE_PATH)

    experiment_meta = {
        "file": path.replace("\\", "/"),
        "role": "baseline" if is_baseline else "candidate",
        "label": meta.get("label") or "",
        "prompt_hash": meta.get("prompt_hash"),
        "top_k": meta.get("top_k"),
        "fetch_k": meta.get("fetch_k"),
        "git_sha": meta.get("git_sha"),
        "created_at": meta.get("created_at"),
        "suite_seconds": meta.get("suite_seconds"),
        "verdict_vs_baseline": verdict or "n/a",
    }
    project = client.create_project(
        project_name=name,
        reference_dataset_id=dataset.id,
        description=meta.get("label") or path,
        metadata=experiment_meta,
    )

    # One run per experiment: the suite run itself. The raw metric values go in
    # its outputs; the statuses from compare.py go alongside them.
    # LangSmith only accepts a run start_time within 24 h of now, so older
    # snapshots are stamped with the upload time; the real time is kept in the
    # metadata (created_at) and in the experiment name.
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=float(meta.get("suite_seconds") or 0))
    run_id = uuid.uuid4()
    client.create_run(
        id=run_id,
        name="eval_suite",
        run_type="chain",
        # by id, not name: after --force deletes a project, deletion finishes in
        # the background and a run sent by name can land in the dying project
        session_id=project.id,
        reference_example_id=example.id,
        inputs=example.inputs,
        outputs={"metrics": metrics, "statuses": statuses,
                 "verdict_vs_baseline": verdict or "n/a"},
        extra={"metadata": experiment_meta},
        start_time=start,
        end_time=end,
    )
    client.flush()   # the run must exist before feedback is attached to it

    # Every metric -> one feedback score (the comparison view charts these).
    n = 0
    for mid, value in sorted(metrics.items()):
        score = _score(value)
        if score is None:
            continue
        comment = f"status vs baseline: {statuses[mid]}" if mid in statuses else None
        client.create_feedback(run_id=run_id, trace_id=run_id, session_id=project.id,
                               key=mid, score=score, comment=comment)
        n += 1
    if verdict:
        client.create_feedback(run_id=run_id, trace_id=run_id, session_id=project.id,
                               key="verdict", score=VERDICT_SCORE[verdict],
                               value=verdict, comment="PASS=1, REVIEW=0.5, FAIL=0")
        n += 1

    client.update_project(project.id, end_time=end)
    shown = f"  verdict={verdict}" if verdict else ""
    print(f"  ok    {name}  ({n} scores){shown}")
    return project


# ============================================================
# 3. LOG MANY
# ============================================================
def log_snapshots(paths, force=False):
    from langsmith import Client   # imported here so the module loads without it

    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is not set (add it to .env).")

    client = Client()
    dataset, example = get_or_create_dataset(client)
    baseline = load(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else None

    print(f"Logging {len(paths)} snapshot(s) to LangSmith dataset '{DATASET_NAME}'...")
    for path in paths:
        try:
            log_snapshot(client, path, dataset, example, baseline, force=force)
        except Exception as e:                       # keep going with the rest
            print(f"  FAIL  {path}: {type(e).__name__}: {e}")
    client.flush()

    url = getattr(dataset, "url", None)
    if url:
        print(f"\nCompare experiments here: {url}")


def main():
    ap = argparse.ArgumentParser(description="Log eval snapshots to LangSmith.")
    ap.add_argument("paths", nargs="*",
                    help=f"snapshot files or globs (default: {DEFAULT_GLOB})")
    ap.add_argument("--force", action="store_true",
                    help="replace experiments that were already logged")
    args = ap.parse_args()

    # Expand globs ourselves: PowerShell doesn't.
    patterns = args.paths or [DEFAULT_GLOB]
    paths = sorted({p for pat in patterns for p in (glob.glob(pat) or [pat])})
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"not found: {', '.join(missing)}")
    log_snapshots(paths, force=args.force)


if __name__ == "__main__":
    main()
