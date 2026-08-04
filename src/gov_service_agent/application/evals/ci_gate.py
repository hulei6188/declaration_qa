"""通过生产API运行评测并把门禁结果转换为CI退出码。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("GOV_SERVICE_AGENT_BASE_URL", ""))
    parser.add_argument("--dataset-id", default=os.getenv("GOV_SERVICE_AGENT_EVAL_DATASET_ID", ""))
    parser.add_argument("--baseline-run-id", default=os.getenv("GOV_SERVICE_AGENT_BASELINE_RUN_ID", ""))
    parser.add_argument("--api-key", default=os.getenv("GOV_SERVICE_AGENT_API_KEY", ""))
    parser.add_argument("--git-commit", default=os.getenv("GITHUB_SHA", ""))
    parser.add_argument("--use-ragas", action="store_true")
    parser.add_argument("--run-retrieval-ablation", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    if not args.base_url or not args.dataset_id or not args.api_key:
        raise SystemExit("base URL, dataset ID and API key are required")
    headers = {"X-Gov-Service-Agent-API-Key": args.api_key, "Content-Type": "application/json"}
    response = requests.post(
        f"{args.base_url.rstrip('/')}/api/declaration-qa/evaluation/datasets/{args.dataset_id}/runs",
        headers=headers,
        json={
            "note": "ci evaluation gate",
            "top_k": 5,
            "use_ragas": args.use_ragas,
            "baseline_run_id": args.baseline_run_id or None,
            "trigger_source": "ci",
            "git_commit": args.git_commit,
            "run_retrieval_ablation": args.run_retrieval_ablation,
        },
        timeout=30,
    )
    response.raise_for_status()
    run = response.json()
    deadline = time.monotonic() + args.timeout
    while run.get("status") in {"queued", "running"}:
        if time.monotonic() >= deadline:
            raise SystemExit("evaluation run timed out")
        time.sleep(5)
        response = requests.get(
            f"{args.base_url.rstrip('/')}/api/declaration-qa/evaluation/runs/{run['run_id']}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        run = response.json()
    print(json.dumps({"run_id": run.get("run_id"), "status": run.get("status"), "metrics": run.get("metrics"), "gate": run.get("gate_result")}, ensure_ascii=False, indent=2))
    if run.get("status") != "succeeded" or not (run.get("gate_result") or {}).get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
