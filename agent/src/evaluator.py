"""Ground-truth evaluation with optional CAMEL-based semantic matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from .llm_client import LlmClient
from .structured_outputs import GroundTruthMatch, GroundTruthMatchBatch
from .types import BugFinding


@dataclass
class MatchDetail:
    """Detailed ground-truth match information."""

    predicted_title: str
    predicted_description: str
    match_id: str
    score: float
    rationale: str = ""
    matched: bool = False


@dataclass
class EvaluationResult:
    """Evaluation metrics."""

    precision: float
    recall: float
    matched: int
    total_predicted: int
    total_ground_truth: int
    details: List[MatchDetail] = field(default_factory=list)


class Evaluator:
    """Compares reported bugs with ground truth."""

    def __init__(
        self,
        ground_truth_path: str,
        match_threshold: float = 0.65,
        llm_client: Optional[LlmClient] = None,
        batch: bool = False,
    ) -> None:
        self._ground_truth_path = Path(ground_truth_path)
        self._match_threshold = match_threshold
        self._batch = batch
        self._ground_truth = self._load_ground_truth()
        self._match_agent = (
            llm_client.create_task_agent(
                system_prompt="You are a QA evaluator.",
                agent_id="ground-truth-evaluator",
            )
            if llm_client is not None
            else None
        )

    def evaluate(self, bugs: List[BugFinding]) -> EvaluationResult:
        """Evaluate predicted bugs against the ground-truth set."""
        if not self._ground_truth:
            return EvaluationResult(0.0, 0.0, 0, len(bugs), 0, [])

        if self._match_agent is None:
            return self._evaluate_with_similarity(bugs)
        if self._batch:
            return self._evaluate_with_camel_batch(bugs)
        return self._evaluate_with_camel(bugs)

    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if not self._ground_truth_path.exists():
            return []
        with open(self._ground_truth_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        bugs = data.get("bugs", [])
        if not isinstance(bugs, list):
            return []
        return [
            self._normalize_truth_entry(item)
            for item in bugs
            if isinstance(item, dict)
        ]

    def _evaluate_with_similarity(self, bugs: List[BugFinding]) -> EvaluationResult:
        matched = 0
        details: List[MatchDetail] = []
        used_truth_indices = set()
        for bug in bugs:
            match_index, score = self._best_match_index(bug, used_truth_indices)
            is_match = match_index is not None
            if is_match:
                matched += 1
                used_truth_indices.add(match_index)
            truth = self._ground_truth[match_index] if match_index is not None else {}
            details.append(
                MatchDetail(
                    predicted_title=bug.title,
                    predicted_description=bug.description,
                    match_id=str(truth.get("id", "")),
                    score=score,
                    rationale="sequence_matcher",
                    matched=is_match,
                )
            )
        return self._build_result(matched, bugs, details)

    def _evaluate_with_camel(self, bugs: List[BugFinding]) -> EvaluationResult:
        matched_truth = set()
        used_truth_indices = set()
        details: List[MatchDetail] = []
        for bug in bugs:
            response = self._match_agent.run(
                self._build_prompt(bug, self._ground_truth),
                response_format=GroundTruthMatch,
            )
            payload = response.parsed
            if payload is None:
                fallback_detail, match_index = self._similarity_detail(
                    bug,
                    used_truth_indices=used_truth_indices,
                    rationale=response.error or "invalid_structured_output",
                )
                details.append(fallback_detail)
                if match_index is not None and fallback_detail.matched:
                    used_truth_indices.add(match_index)
                    matched_truth.add(str(self._ground_truth[match_index].get("id", "")))
                continue
            is_match = (
                bool(payload.match_id)
                and payload.score >= self._match_threshold
                and payload.match_id not in matched_truth
            )
            truth_index = self._truth_index_for_id(payload.match_id)
            if is_match:
                matched_truth.add(payload.match_id)
                if truth_index is not None:
                    used_truth_indices.add(truth_index)
            details.append(
                MatchDetail(
                    predicted_title=bug.title,
                    predicted_description=bug.description,
                    match_id=payload.match_id,
                    score=float(payload.score),
                    rationale=payload.rationale,
                    matched=is_match,
                )
            )
        matched = sum(1 for item in details if item.matched)
        return self._build_result(matched, bugs, details, len(matched_truth))

    def _evaluate_with_camel_batch(
        self, bugs: List[BugFinding]
    ) -> EvaluationResult:
        matched_truth: set[str] = set()
        used_truth_indices: set[int] = set()
        details: List[MatchDetail] = []

        response = self._match_agent.run(
            self._build_batch_prompt(bugs, self._ground_truth),
            response_format=GroundTruthMatchBatch,
        )
        payload = response.parsed
        if payload is None or not payload.matches:
            # Fallback to similarity for the entire batch
            return self._evaluate_with_similarity(bugs)

        # Build lookup by bug_index
        match_items = {item.bug_index: item for item in payload.matches}

        for idx, bug in enumerate(bugs):
            item = match_items.get(idx)
            if item is None:
                # Fallback for this specific bug
                fallback_detail, match_index = self._similarity_detail(
                    bug,
                    used_truth_indices=used_truth_indices,
                    rationale="missing_batch_item",
                )
                details.append(fallback_detail)
                if match_index is not None and fallback_detail.matched:
                    used_truth_indices.add(match_index)
                    matched_truth.add(str(self._ground_truth[match_index].get("id", "")))
                continue

            is_match = (
                bool(item.match_id)
                and item.score >= self._match_threshold
                and item.match_id not in matched_truth
            )
            truth_index = self._truth_index_for_id(item.match_id)
            if is_match:
                matched_truth.add(item.match_id)
                if truth_index is not None:
                    used_truth_indices.add(truth_index)
            details.append(
                MatchDetail(
                    predicted_title=bug.title,
                    predicted_description=bug.description,
                    match_id=item.match_id,
                    score=float(item.score),
                    rationale=item.rationale,
                    matched=is_match,
                )
            )
        matched = sum(1 for d in details if d.matched)
        return self._build_result(matched, bugs, details, len(matched_truth))

    def _similarity_detail(
        self,
        bug: BugFinding,
        *,
        used_truth_indices: set[int],
        rationale: str,
    ) -> tuple[MatchDetail, Optional[int]]:
        match_index, score = self._best_match_index(bug, used_truth_indices)
        truth = self._ground_truth[match_index] if match_index is not None else {}
        matched = match_index is not None
        detail = MatchDetail(
            predicted_title=bug.title,
            predicted_description=bug.description,
            match_id=str(truth.get("id", "")),
            score=score,
            rationale=f"similarity_fallback:{rationale}",
            matched=matched,
        )
        return detail, match_index

    def _build_result(
        self,
        matched: int,
        bugs: List[BugFinding],
        details: List[MatchDetail],
        unique_truth_matches: Optional[int] = None,
    ) -> EvaluationResult:
        precision = matched / len(bugs) if bugs else 0.0
        recall_denominator = len(self._ground_truth)
        recall_numerator = (
            unique_truth_matches if unique_truth_matches is not None else matched
        )
        recall = (
            recall_numerator / recall_denominator if recall_denominator else 0.0
        )
        return EvaluationResult(
            precision=precision,
            recall=recall,
            matched=matched,
            total_predicted=len(bugs),
            total_ground_truth=recall_denominator,
            details=details,
        )

    def _best_match_index(
        self,
        bug: BugFinding,
        used_indices: set[int],
    ) -> tuple[Optional[int], float]:
        best_score = 0.0
        best_index = None
        for index, truth in enumerate(self._ground_truth):
            if index in used_indices:
                continue
            score = self._similarity(bug, truth)
            if score > best_score:
                best_score = score
                best_index = index
        if best_score >= self._match_threshold:
            return best_index, best_score
        return None, best_score

    def _truth_index_for_id(self, match_id: str) -> Optional[int]:
        if not match_id:
            return None
        for index, truth in enumerate(self._ground_truth):
            if str(truth.get("id", "")) == match_id:
                return index
        return None

    @staticmethod
    def _similarity(bug: BugFinding, truth: Dict[str, Any]) -> float:
        text = f"{bug.title} {bug.description}"
        truth_text = Evaluator._truth_text(truth)
        return SequenceMatcher(None, text, truth_text).ratio()

    @staticmethod
    def _build_prompt(
        bug: BugFinding,
        ground_truth: List[Dict[str, Any]],
    ) -> str:
        items = [
            (
                f"- {item['id']} | bug_type={item.get('bug_type', '')} | "
                f"difficulty={item.get('difficulty', '')} | "
                f"minimal_reproduction={'; '.join(item.get('minimal_reproduction', []))} | "
                f"observed_fault={item.get('observed_fault', '')}"
            )
            for item in ground_truth
        ]
        gt_text = "\n".join(items)
        return (
            "Evaluate whether the predicted bug matches one of the ground-truth bugs.\n"
            "Return structured output with fields: match_id, score, rationale.\n"
            "score must be between 0.0 and 1.0. If there is no match, use an empty match_id.\n\n"
            f"Predicted bug:\nTitle: {bug.title}\nDescription: {bug.description}\n\n"
            f"Ground truth bugs:\n{gt_text}\n"
        )

    @staticmethod
    def _build_batch_prompt(
        bugs: List[BugFinding],
        ground_truth: List[Dict[str, Any]],
    ) -> str:
        gt_items = [
            (
                f"- {item['id']} | bug_type={item.get('bug_type', '')} | "
                f"difficulty={item.get('difficulty', '')} | "
                f"minimal_reproduction={'; '.join(item.get('minimal_reproduction', []))} | "
                f"observed_fault={item.get('observed_fault', '')}"
            )
            for item in ground_truth
        ]
        gt_text = "\n".join(gt_items)

        pred_items = [
            f"{idx}. Title: {bug.title}\n   Description: {bug.description}"
            for idx, bug in enumerate(bugs)
        ]
        pred_text = "\n".join(pred_items)

        return (
            "Evaluate whether each predicted bug matches one of the ground-truth bugs.\n"
            "Return structured output with a list of matches.\n"
            "For each predicted bug, provide: bug_index (0-based), match_id (ground-truth id or empty string), score (0.0-1.0), rationale.\n\n"
            "A match means the predicted bug belongs to the same fault class as a ground-truth bug. A fault class is defined by the root cause and the type of system failure, not by the exact user action, object, or output symptom.\n"
            "Do not subdivide fault classes by the specific object involved (e.g., scroll vs ladder vs generic portable item) or by the exact visible symptom (e.g., stale text remaining vs updated text missing).\n"
            "For example, a failure where 'backend state changes are not reflected in displayed text' can manifest as stale text remaining visible OR as updated text failing to appear; both belong to the same fault class if the root cause is the same.\n"
            "Do not match predictions that are merely about the same broad topic.\n"
            "Do not match predictions that are purely speculative or hedged without describing a concrete buggy observation.\n"
            "Score must be between 0.0 and 1.0. Use scores >= 0.6 for genuine matches and lower scores for related but non-matching issues.\n"
            "Each ground-truth bug may be matched at most once. "
            "If several predicted bugs correspond to the same ground-truth bug, match the clearest one and use an empty match_id for the others.\n"
            "Do not leave all predictions unmatched simply because they are duplicated.\n"
            "If a predicted bug does not match any ground-truth bug, use an empty match_id.\n\n"
            f"Predicted bugs:\n{pred_text}\n\n"
            f"Ground truth bugs:\n{gt_text}\n"
        )

    @staticmethod
    def _normalize_truth_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
        minimal_reproduction = payload.get("minimal_reproduction") or payload.get(
            "test_steps", []
        )
        if isinstance(minimal_reproduction, str):
            minimal_reproduction = [minimal_reproduction.strip()] if minimal_reproduction.strip() else []
        elif isinstance(minimal_reproduction, list):
            minimal_reproduction = [
                str(step).strip() for step in minimal_reproduction if str(step).strip()
            ]
        else:
            minimal_reproduction = []

        observed_fault = str(
            payload.get("observed_fault")
            or payload.get("actual_behavior")
            or payload.get("description")
            or ""
        ).strip()
        title = str(payload.get("title") or "").strip()
        description = str(payload.get("description") or "").strip()

        if not title:
            title = observed_fault or str(payload.get("id", "")).strip()
        if not description:
            description = observed_fault

        return {
            "id": str(payload.get("id", "")).strip(),
            "bug_type": str(payload.get("bug_type", "")).strip(),
            "difficulty": str(payload.get("difficulty", "")).strip(),
            "minimal_reproduction": minimal_reproduction,
            "observed_fault": observed_fault,
            "title": title,
            "description": description,
        }

    @staticmethod
    def _truth_text(truth: Dict[str, Any]) -> str:
        parts = [
            str(truth.get("title", "")).strip(),
            str(truth.get("description", "")).strip(),
            str(truth.get("bug_type", "")).strip(),
            str(truth.get("difficulty", "")).strip(),
            " ".join(str(step).strip() for step in truth.get("minimal_reproduction", [])),
            str(truth.get("observed_fault", "")).strip(),
        ]
        return " ".join(part for part in parts if part)
