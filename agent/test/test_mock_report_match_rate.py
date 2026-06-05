"""Print the offline similarity baseline for the mock Dark Castle report.

Supports two evaluation modes:
- Similarity (default): uses SequenceMatcher against ground truth.
- LLM (optional): uses a CAMEL task agent for semantic matching.

To enable LLM mode, set the environment variable:
    USE_LLM_EVAL=1

Required environment variables for LLM mode (read by LlmClient):
    API_KEY, MODEL_NAME, BASE_URL (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load repo-root .env so API_KEY / MODEL_NAME / BASE_URL are available
_repo_root = Path(ROOT_DIR).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from gbqa.env import load_root_dotenv

load_root_dotenv()

from src.evaluator import Evaluator
from src.llm_client import LlmClient
from src.types import BugFinding


def load_bug_findings(report_path: Path) -> list[BugFinding]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    findings = []
    for item in payload.get("bugs", []):
        findings.append(
            BugFinding(
                title=str(item.get("title", "")).strip(),
                description=str(item.get("description", "")).strip(),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                evidence=item.get("evidence", {}) or {},
                tags=item.get("tags", []) or [],
            )
        )
    return findings


def create_llm_client() -> LlmClient:
    """Instantiate an LlmClient from environment variables."""
    return LlmClient(config={})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate mock Dark Castle report against ground truth."
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=os.getenv("USE_LLM_EVAL", "").lower() in ("1", "true", "yes"),
        help="Use LLM-based semantic matching instead of SequenceMatcher. "
             "Can also be enabled via USE_LLM_EVAL=1 env var.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.65,
        help="Similarity threshold for a positive match (default: 0.65).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Send all predicted bugs in a single LLM call instead of one-by-one. "
             "Only applies when --use-llm is enabled.",
    )
    args = parser.parse_args()

    report_path = (
        Path(ROOT_DIR) / "test" / "mock_reports" / "dark-castle" / "report.json"
    )
    ground_truth_path = (
        Path(ROOT_DIR)
        / ".."
        / "gbqa"
        / "tasks"
        / "dark-castle"
        / "bugs"
        / "dark-castle.json"
    ).resolve()

    llm_client = create_llm_client() if args.use_llm else None
    evaluator = Evaluator(
        str(ground_truth_path),
        match_threshold=args.match_threshold,
        llm_client=llm_client,
        batch=args.batch,
    )
    result = evaluator.evaluate(load_bug_findings(report_path))

    output = {
        "report": str(report_path),
        "ground_truth": str(ground_truth_path),
        "evaluator": "llm" if args.use_llm else "similarity",
        "batch": args.batch,
        "precision": result.precision,
        "recall": result.recall,
        "matched": result.matched,
        "predicted_total": result.total_predicted,
        "ground_truth_total": result.total_ground_truth,
        "details": [detail.__dict__ for detail in result.details],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    assert result.total_ground_truth == 3
    assert result.total_predicted == 3


if __name__ == "__main__":
    main()
