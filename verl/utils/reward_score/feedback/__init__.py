from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse


def _normalize_data_source(data_source: str) -> str:
    aliases = {
        "TianHongZXY/MATH": "math",
    }
    return aliases.get(data_source, data_source)


def _normalize_ground_truth(ground_truth):
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        return ground_truth["target"]
    return ground_truth


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict = None,
) -> dict:
    # Normalize dataset-specific metadata into the canonical reward routing format.
    data_source = _normalize_data_source(data_source)
    ground_truth = _normalize_ground_truth(ground_truth)

    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
