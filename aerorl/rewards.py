from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(slots=True)
class RewardContext:
    prompt: str
    response: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardResult:
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


class RewardFunction(Protocol):
    name: str

    def __call__(self, context: RewardContext) -> RewardResult:
        ...


def _safe_token_count(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())


def _clip_score(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(slots=True)
class FormatReward:
    require_json: bool = False
    regex_pattern: str | None = None
    name: str = "format"

    def __call__(self, context: RewardContext) -> RewardResult:
        score = 1.0
        details: dict[str, Any] = {}

        if self.require_json:
            try:
                json.loads(context.response)
                details["json_valid"] = True
            except json.JSONDecodeError:
                details["json_valid"] = False
                score -= 1.3

        if self.regex_pattern:
            matched = re.search(self.regex_pattern, context.response) is not None
            details["regex_matched"] = matched
            if not matched:
                score -= 0.5

        return RewardResult(name=self.name, score=_clip_score(score), details=details)


@dataclass(slots=True)
class VerifierReward:
    case_sensitive: bool = False
    name: str = "verifier"

    def __call__(self, context: RewardContext) -> RewardResult:
        if context.reference is None:
            return RewardResult(name=self.name, score=0.0, details={"reason": "missing_reference"})

        response = context.response.strip()
        reference = context.reference.strip()
        if not self.case_sensitive:
            response = response.lower()
            reference = reference.lower()

        exact = response == reference
        contains = reference in response if reference else False
        score = 1.0 if exact else (0.5 if contains else -0.5)
        return RewardResult(
            name=self.name,
            score=score,
            details={"exact_match": exact, "contains_reference": contains},
        )


@dataclass(slots=True)
class GroundingReward:
    name: str = "grounding"

    def __call__(self, context: RewardContext) -> RewardResult:
        evidence_entities = context.metadata.get("evidence_entities", [])
        claimed_entities = context.metadata.get("claimed_entities", [])

        evidence_set = {str(item).strip().lower() for item in evidence_entities if str(item).strip()}
        claimed_set = {str(item).strip().lower() for item in claimed_entities if str(item).strip()}

        if not claimed_set:
            return RewardResult(name=self.name, score=0.0, details={"reason": "no_claimed_entities"})

        overlap = len(evidence_set.intersection(claimed_set))
        precision = overlap / max(len(claimed_set), 1)
        score = _clip_score(2.0 * precision - 1.0)
        return RewardResult(
            name=self.name,
            score=score,
            details={"claimed": len(claimed_set), "overlap": overlap, "precision": round(precision, 4)},
        )


@dataclass(slots=True)
class CostReward:
    target_tokens: int = 128
    latency_budget_ms: float = 500.0
    name: str = "cost"

    def __call__(self, context: RewardContext) -> RewardResult:
        token_count = _safe_token_count(context.response)
        latency_ms = float(context.metadata.get("latency_ms", 0.0))

        token_ratio = token_count / max(self.target_tokens, 1)
        latency_ratio = latency_ms / max(self.latency_budget_ms, 1.0)

        penalty = max(0.0, token_ratio - 1.0) + max(0.0, latency_ratio - 1.0)
        score = _clip_score(1.0 - penalty)

        return RewardResult(
            name=self.name,
            score=score,
            details={
                "token_count": token_count,
                "latency_ms": latency_ms,
                "target_tokens": self.target_tokens,
                "latency_budget_ms": self.latency_budget_ms,
            },
        )


@dataclass(slots=True)
class WeightedRewardStack:
    reward_functions: list[RewardFunction]
    weights: dict[str, float]

    def evaluate(self, context: RewardContext) -> dict[str, Any]:
        component_scores: dict[str, float] = {}
        component_details: dict[str, dict[str, Any]] = {}
        total = 0.0

        for reward_fn in self.reward_functions:
            result = reward_fn(context)
            component_scores[result.name] = result.score
            component_details[result.name] = result.details
            total += self.weights.get(result.name, 1.0) * result.score

        return {
            "total_reward": round(total, 6),
            "components": component_scores,
            "details": component_details,
        }


def build_default_reward_stack() -> WeightedRewardStack:
    return build_reward_stack()


def build_reward_stack(
    *,
    weights: dict[str, float] | None = None,
    require_json: bool = False,
    regex_pattern: str | None = None,
    target_tokens: int = 128,
    latency_budget_ms: float = 500.0,
) -> WeightedRewardStack:
    rewards: list[RewardFunction] = [
        VerifierReward(),
        GroundingReward(),
        FormatReward(require_json=require_json, regex_pattern=regex_pattern),
        CostReward(target_tokens=target_tokens, latency_budget_ms=latency_budget_ms),
    ]
    default_weights = {
        "verifier": 0.4,
        "grounding": 0.3,
        "format": 0.2,
        "cost": 0.1,
    }

    merged_weights = dict(default_weights)
    if weights:
        merged_weights.update(weights)
    return WeightedRewardStack(reward_functions=rewards, weights=merged_weights)


def evaluate_records(
    records: Iterable[dict[str, Any]],
    reward_stack: WeightedRewardStack | None = None,
    *,
    pass_threshold: float = 0.3,
    top_k: int = 3,
) -> dict[str, Any]:
    stack = reward_stack or build_default_reward_stack()
    outputs: list[dict[str, Any]] = []

    for record in records:
        context = RewardContext(
            prompt=str(record.get("prompt", "")),
            response=str(record.get("response", "")),
            reference=record.get("reference"),
            metadata=dict(record.get("metadata", {})),
        )
        scored = stack.evaluate(context)
        outputs.append({"id": record.get("id"), "reward": scored})

    if not outputs:
        return {
            "count": 0,
            "average_reward": 0.0,
            "pass_rate": 0.0,
            "component_averages": {},
            "best_examples": [],
            "worst_examples": [],
            "results": [],
        }

    total_scores = [item["reward"]["total_reward"] for item in outputs]
    avg = sum(total_scores) / len(outputs)

    component_names = list(outputs[0]["reward"]["components"].keys())
    component_averages = {
        name: round(sum(item["reward"]["components"][name] for item in outputs) / len(outputs), 6)
        for name in component_names
    }

    passed = sum(1 for value in total_scores if value >= pass_threshold)
    pass_rate = passed / len(outputs)

    sorted_outputs = sorted(outputs, key=lambda item: item["reward"]["total_reward"], reverse=True)
    best_examples = [
        {
            "id": item.get("id"),
            "total_reward": item["reward"]["total_reward"],
            "components": item["reward"]["components"],
        }
        for item in sorted_outputs[:top_k]
    ]
    worst_examples = [
        {
            "id": item.get("id"),
            "total_reward": item["reward"]["total_reward"],
            "components": item["reward"]["components"],
        }
        for item in sorted_outputs[-top_k:]
    ]

    return {
        "count": len(outputs),
        "average_reward": round(avg, 6),
        "pass_rate": round(pass_rate, 6),
        "pass_threshold": pass_threshold,
        "component_averages": component_averages,
        "weights": stack.weights,
        "best_examples": best_examples,
        "worst_examples": worst_examples,
        "results": outputs,
    }
