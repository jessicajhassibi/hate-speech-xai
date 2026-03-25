import numpy as np
import pytest

from hate_speech_xai.src.data.preprocess import get_majority_label, aggregate_rationales, get_post_as_str


class TestGetPostAsStr:
	def test_joins_tokens(self):
		assert (get_post_as_str(["Trump", "is", "an", "asshole", "and", "doesn't", "deserve", "a", "nobel", "price"])
				== "Trump is an asshole and doesn't deserve a nobel price")

	def test_single_token(self):
		assert get_post_as_str(["hello"]) == "hello"

class TestGetMajorityLabel:
	def test_majority(self):
		assert get_majority_label([0, 0, 1]) == 0

	def test_total_majority(self):
		assert get_majority_label([2, 2, 2]) == 2

	def test_all_different_returns_first_in_list(self):
		result = get_majority_label([0, 1, 2])
		assert result in [0, 1, 2]


class TestAggregateRationales:
	def test_normal_label_returns_uniform(self):
		rationales = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
		result = aggregate_rationales(rationales, label=1) # 1 = normal
		expected = np.array([1/3, 1/3, 1/3]) # uniform distribution over 3 tokens
		np.testing.assert_allclose(result, expected) # use assert_allclose for float comparison

	def test_hate_label_values_sum_to_1(self): # same for offensive
		rationales = [[1, 0, 0], [1, 1, 0]] # Averaged: [1.0, 0.5, 0.0]
		result = aggregate_rationales(rationales, label=0)  # 0 = hate speech
		# softmax and the the temperature=5.0 will influence the distribution heavily
		# by design the ground truth rationale should focus most on the most-agreed-upon tokens
		# it will become something like [0.918, 0.075, 0.006]
		assert all(v >= 0 for v in result)
		assert result.sum() == pytest.approx(1.0)
		assert result[0] > result[1] > result[2]  # highest rationale token gets highest weight

	def test_empty_rationales(self):
		result = aggregate_rationales([], label=0)
		assert len(result) == 0

	def test_mismatched_lengths_get_padded(self):
		rationales = [[1, 0], [1, 0, 1]]
		result = aggregate_rationales(rationales, label=0)
		assert len(result) == 3

