import pytest

from hate_speech_xai.src.models.evaluate import get_classification_report


class TestAccuracy:
	# Accuracy = num of correct predictions / total predictions
	# == TP + TN / TP + TN + FP + FN

	def test_all_correct(self):
		y_true = [0, 0, 1, 1, 2, 2]
		y_pred = [0, 0, 1, 1, 2, 2]
		acc, _, _ = get_classification_report(y_true, y_pred)
		assert acc == 1.0

	def test_some_correct(self):
		y_true = [0, 0, 1, 1]
		y_pred = [0, 0, 2, 2]
		acc, _, _ = get_classification_report(y_true, y_pred)
		assert acc == 0.5

	def test_all_different(self):
		y_true = [0, 0, 1, 1, 2, 2]
		y_pred = [1, 1, 2, 2, 0, 0]
		acc, _, _ = get_classification_report(y_true, y_pred)
		assert acc == 0.0


class TestPrecisionAndRecall:
	# Precision = TP / (TP + FP) —> of all predicted as class X, how many are actually X
	# Recall = TP / (TP + FN) —> of all actual class X, how many were found

	def test_high_precision_low_recall(self):
		# Hate speech is predicted only once which is correct -> precision=1.0
		# But there are 3 actual Hate speech posts! -> recall=1/3
		y_true = [0, 0, 0, 1, 1, 2]
		y_pred = [0, 1, 1, 1, 1, 2]
		_, _, report = get_classification_report(y_true, y_pred)
		assert report["Hate speech"]["precision"] == 1.0
		assert report["Hate speech"]["recall"] == pytest.approx(1/3)

	def test_low_precision_high_recall(self):
		# Hate speech is always predicted -> recall=1.0
		# But only 2 out of 6 are really Hate speech -> precision=2/6
		y_true = [0, 0, 1, 1, 2, 2]
		y_pred = [0, 0, 0, 0, 0, 0]
		_, _, report = get_classification_report(y_true, y_pred)
		assert report["Hate speech"]["recall"] == 1.0
		assert report["Hate speech"]["precision"] == pytest.approx(1/3)


class TestF1Score:
	# F1 (macro) is the average of the per-class F1 scores
	# Per-class F1 = 2 * (precision * recall) / (precision + recall)

	def test_correct_for_all_classes(self):
		y_true = [0, 1, 2, 0, 1, 2]
		y_pred = [0, 1, 2, 0, 1, 2]
		_, f1, _ = get_classification_report(y_true, y_pred)
		assert f1 == 1.0

	def test_only_normal_correct(self):
		# Hate speech and Offensive are never predicted
		y_true = [0, 1, 2]
		y_pred = [1, 1, 1]
		_, f1, report = get_classification_report(y_true, y_pred)
		# Normal: precision=1/3, recall=1/1=1.0 -> F1=2*(1/3*1)/(1/3+1)=0.5
		# Hate speech: never predicted -> F1=0
		# Offensive: never predicted -> F1=0
		# Macro F1 = (0 + 0.5 + 0) / 3
		assert f1 == pytest.approx(0.5/3)
