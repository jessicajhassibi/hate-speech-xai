import json
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from hate_speech_xai.config import SAVED_MODELS_DIR, LABELS

label_names = [LABELS[i] for i in sorted(LABELS.keys())]


def load_evaluation_results(source: Path = SAVED_MODELS_DIR) -> tuple[list[int], list[int]] | None:
	"""Load saved test evaluation results (y_true, y_pred) from JSON."""
	eval_path = source / "test_evaluation.json"
	if not eval_path.exists():
		return None
	with open(eval_path) as f:
		eval_results = json.load(f)
	return eval_results["y_true"], eval_results["y_pred"]


def load_xai_evaluation_results(source: Path = SAVED_MODELS_DIR) -> list[dict] | None:
	"""Load saved XAI evaluation results from JSON."""
	eval_path = source / "xai_evaluation.json"
	if not eval_path.exists():
		return None
	with open(eval_path) as f:
		return json.load(f)


def get_classification_report(y_true: list[int], y_pred: list[int]) -> tuple[float, float, dict]:
	"""Return overall metrics and per-class classification report."""
	acc = accuracy_score(y_true, y_pred)
	f1_macro = f1_score(y_true, y_pred, average="macro")
	report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
	return acc, f1_macro, report


def plot_confusion_matrix(y_true: list[int], y_pred: list[int]) -> Figure:
	"""Create and return a confusion matrix figure."""
	cm = confusion_matrix(y_true, y_pred)
	fig, ax = plt.subplots(figsize=(2.8, 2.4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax,
				annot_kws={"size": 8})
	ax.set_xlabel("Predicted", fontsize=8)
	ax.set_ylabel("Actual", fontsize=8)
	ax.tick_params(labelsize=6)
	colorbar = ax.collections[0].colorbar
	colorbar.ax.tick_params(labelsize=6)
	fig.tight_layout()
	return fig