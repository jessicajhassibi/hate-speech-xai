import numpy as np

from hate_speech_xai.src.models.evaluate_xai import top_k_overlap


class TestTopKOverlap:
	"""Checks if the fraction of the top-k important tokens that are also in the ground truth rationale is correctly calculated."""

	def test_perfect_overlap(self):
		rationale = np.array([0.0, 1.0, 1.0]) # ground truth: tokens 1 and 2 are important
		importance = np.array([0.1, 0.9, 0.8]) # calculated: most k=2 important tokens are 1 and 2
		result = top_k_overlap(importance, rationale)
		assert result == 1.0 # from k=2 actual important tokens, both are in the top-k calculated important tokens

	def test_no_overlap(self):
		rationale = np.array([0.0, 0.0, 1.0])
		importance = np.array([0.9, 0.0, 0.0])
		result = top_k_overlap(importance, rationale) # k will become 1
		assert result == 0.0

	def test_partial_overlap(self):
		rationale = np.array([0.0, 1.0, 0.0, 1.0])
		importance = np.array([0.9, 0.8, 0.1, 0.0])
		result = top_k_overlap(importance, rationale) # k will become 2
		assert result == 0.5

	def test_empty_importance(self):
		rationale = np.array([1.0])
		importance = np.array([])
		result = top_k_overlap(importance, rationale)
		assert result is None

	def test_empty_rationale(self):
		rationale = np.array([])
		importance = np.array([1.0])
		result = top_k_overlap(importance, rationale)
		assert result is None

	def test_uniform_rationale_returns_none(self):
		# All equal values -> k=0 (none above mean) -> None
		# for normal posts
		importance = np.array([0.5, 0.5, 0.5])
		rationale = np.array([0.5, 0.5, 0.5])
		result = top_k_overlap(importance, rationale)
		assert result is None
