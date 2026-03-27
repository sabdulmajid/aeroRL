from aerorl import (
    RewardContext,
    WeightedRewardStack,
    VerifierReward,
    GroundingReward,
    FormatReward,
    CostReward,
    build_default_reward_stack,
    build_reward_stack,
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

    summary = evaluate_records(records, pass_threshold=0.0, top_k=1)

    assert summary["count"] == 2
    assert len(summary["results"]) == 2
    assert isinstance(summary["average_reward"], float)
    assert "component_averages" in summary
    assert "pass_rate" in summary
    assert len(summary["best_examples"]) == 1
    assert len(summary["worst_examples"]) == 1


def test_build_reward_stack_customization() -> None:
    stack = build_reward_stack(
        weights={"verifier": 1.0, "grounding": 0.0, "format": 0.0, "cost": 0.0},
        require_json=True,
        regex_pattern=r"^\{.*\}$",
        target_tokens=16,
        latency_budget_ms=200,
    )

    assert stack.weights["verifier"] == 1.0
    context = RewardContext(
        prompt="p",
        response='{"answer": "ok"}',
        reference='{"answer": "ok"}',
        metadata={"latency_ms": 10, "evidence_entities": ["ok"], "claimed_entities": ["ok"]},
    )
    scored = stack.evaluate(context)
    assert scored["components"]["format"] >= 0.0


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


def test_verifier_accepts_multiple_references() -> None:
    verifier = VerifierReward()
    context = RewardContext(
        prompt="p",
        response="april 15, 2014",
        reference=["april 15,2014", "april 15, 2014"],
        metadata={},
    )

    scored = verifier(context)

    assert scored.score == 1.0
    assert scored.details["matched_reference"] == "april 15, 2014"
    assert scored.details["reference_count"] == 2


def test_grounding_normalizes_common_surface_form_variants() -> None:
    grounding = GroundingReward()
    cases = [
        ("42", "42%"),
        ("Mr. Alm", "Mr. Alm."),
        ("next 2-3 weeks", "within the next 2-3 weeks"),
    ]

    for evidence, claimed in cases:
        context = RewardContext(
            prompt="p",
            response=claimed,
            reference=None,
            metadata={"evidence_entities": [evidence], "claimed_entities": [claimed]},
        )
        assert grounding(context).score == 1.0
