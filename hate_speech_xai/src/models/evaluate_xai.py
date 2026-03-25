import numpy as np
from datasets import Dataset

from hate_speech_xai.src.data.preprocess import get_majority_label, aggregate_rationales, get_post_as_str
from hate_speech_xai.src.models.explain import EXPLANATION_METHODS


def top_k_overlap(importance: np.ndarray, rationale: np.ndarray, k: int | None = None) -> float | None:
	"""Fraction of the top-k important tokens that are also in the ground truth rationale.
	k defaults to the number of rationale tokens (tokens with above-average rationale value).
	"""
	if len(importance) == 0 or len(rationale) == 0:
		return None

	min_len = min(len(importance), len(rationale))
	importance = importance[:min_len]
	rationale = rationale[:min_len]

	# Number of "important" tokens in ground truth
	if k is None:
		k = int((rationale > rationale.mean()).sum())
	if k == 0:
		return None

	top_k_indices = set(np.argsort(importance)[-k:])
	rationale_indices = set(np.argsort(rationale)[-k:])

	overlap = len(top_k_indices & rationale_indices)
	return overlap / k


def evaluate_xai_on_dataset(dataset: Dataset, method_name: str, method_fn: callable, max_samples: int | None = None) -> dict:
	"""Evaluates the passed XAI method on a dataset and returns per-metric averages."""
	top_k_scores = []

	samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

	for i in range(len(samples)):
		post = samples[i]
		label = get_majority_label(post["annotators"]["label"])

		# Skip normal posts (uniform rationale, not meaningful to evaluate)
		if label == 1:
			continue

		rationale = aggregate_rationales(post["rationales"], label)
		text = get_post_as_str(post["post_tokens"])

		if len(rationale) == 0:
			continue

		importance = method_fn(text)

		tk = top_k_overlap(importance, rationale)
		if tk is not None:
			top_k_scores.append(tk)

	return {
		"method": method_name,
		"top_k_overlap": np.mean(top_k_scores) if top_k_scores else None,
		"n_samples": len(top_k_scores),
	}


def evaluate_all_xai_methods(dataset: Dataset, max_samples: int | None = None) -> list[dict]:
	results = []
	for method_name, method_fn in EXPLANATION_METHODS.items():
		print(f"Evaluating {method_name}...")
		result = evaluate_xai_on_dataset(dataset, method_name, method_fn, max_samples)
		results.append(result)
		print(f"  Top-k overlap: {result['top_k_overlap']:.4f}")
		print(f"  Evaluated on {result['n_samples']} samples")
	return results