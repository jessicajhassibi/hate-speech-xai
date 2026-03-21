from datasets import Dataset

from hate_speech_xai.config import LABELS
from hate_speech_xai.src.data.preprocess import get_majority_label

def count_num_of_annotators(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> int:
	"""Count the total number of unique annotators in the dataset by collecting all annotator IDs from each split."""
	all_annotator_ids = set()
	for ds in [train_ds, val_ds, test_ds]:
		for post in ds:
			annotator_ids = post['annotators']['annotator_id']
			all_annotator_ids.update(annotator_ids)
	total_annotators = len(all_annotator_ids)
	return total_annotators

def compute_label_distribution(splits: dict) -> list[dict]:
	"""Compute majority-vote label counts per split.
	Returns a list of dicts with 'Split' and 'Label' keys.
	"""
	rows = []
	for split_name, ds in splits.items():
		for i in range(len(ds)):
			label_id = get_majority_label(ds[i]["annotators"]["label"])
			rows.append({"Split": split_name, "Label": LABELS[label_id]})
	return rows