from aerorl import (
    RewardContext,
    WeightedRewardStack,
    VerifierReward,
    GroundingReward,
    FormatReward,
    CostReward,
    build_default_reward_stack,
    evaluate_records,
)


def test_default_reward_stack_scoring() -> None:
    stack = build_default_reward_stack()
    context = RewardContext(
        prompt="What text is on the sign?",
        response="stop",
        reference="stop",
        metadata={"evidence_entities": ["stop"], "claimed_entities": ["stop"], "latency_ms": 120},
    )

    scored = stack.evaluate(context)

    assert "total_reward" in scored
    assert scored["components"]["verifier"] == 1.0
    assert scored["components"]["grounding"] == 1.0


def test_custom_stack_penalizes_bad_format_and_cost() -> None:
    stack = WeightedRewardStack(
        reward_functions=[FormatReward(require_json=True), CostReward(target_tokens=2, latency_budget_ms=10)],
        weights={"format": 0.7, "cost": 0.3},
    )
    context = RewardContext(
        prompt="return json",
        response="not-json and too many words",
        reference=None,
        metadata={"latency_ms": 40},
    )

    scored = stack.evaluate(context)

    assert scored["components"]["format"] < 0
    assert scored["components"]["cost"] < 1.0


def test_evaluate_records_summary() -> None:
    records = [
        {
            "id": "r1",
            "prompt": "p1",
            "response": "answer",
            "reference": "answer",
            "metadata": {"evidence_entities": ["answer"], "claimed_entities": ["answer"], "latency_ms": 30},
        },
        {
            "id": "r2",
            "prompt": "p2",
            "response": "wrong",
            "reference": "right",
            "metadata": {"evidence_entities": ["right"], "claimed_entities": ["wrong"], "latency_ms": 800},
        },
    ]

    summary = evaluate_records(records)

    assert summary["count"] == 2
    assert len(summary["results"]) == 2
    assert isinstance(summary["average_reward"], float)


def test_individual_rewards() -> None:
    verifier = VerifierReward()
    grounding = GroundingReward()

    context = RewardContext(
        prompt="p",
        response="red car",
        reference="red car",
        metadata={"evidence_entities": ["car", "red"], "claimed_entities": ["car", "red"]},
    )

    assert verifier(context).score == 1.0
    assert grounding(context).score == 1.0
