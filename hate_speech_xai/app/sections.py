import random
import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from hate_speech_xai.app.styling import render_label_badge, render_rationale
from hate_speech_xai.config import MODEL_NAME, LABELS
from hate_speech_xai.src.data.preprocess import get_majority_label, get_post_as_str, aggregate_rationales
from hate_speech_xai.src.data.analyze import compute_label_distribution
from hate_speech_xai.src.models.evaluate import (
	load_evaluation_results, load_xai_evaluation_results,
	get_classification_report, plot_confusion_matrix,
)
from hate_speech_xai.src.models.evaluate_xai import top_k_overlap
from hate_speech_xai.src.models.predict import predict_label
from hate_speech_xai.src.models.explain import EXPLANATION_METHODS

label_names = [LABELS[i] for i in sorted(LABELS.keys())]


@st.cache_data
def _get_label_distribution(_splits: dict) -> pd.DataFrame:
	return pd.DataFrame(compute_label_distribution(_splits))


def dataset_explorer(splits: dict) -> None:
	st.subheader("HateXplain Dataset Explorer")
	st.write("We will explore the HateXplain dataset, a benchmark for explainable hate speech detection.")

	st.markdown("#### Class Distribution")
	st.info("The dataset consists of posts, each labeled by 3 annotators as hate speech, offensive, or normal."
			" It is split into 3 parts: Train, Validation, and Test.")
	st.expander("What is Hate Speech?").markdown(
		"A definition by [Britannica](https://www.britannica.com/topic/hate-speech): *\"Hate speech is speech or expression that denigrates a person or persons on the basis of (alleged) membership "
		"in a social group identified by attributes such as race, ethnicity, gender, sexual orientation, religion, age, "
		"physical or mental disability, and others.\"*\n\n"
		"Regarding the distinction from offensive speech, [DW Academy](https://akademie.dw.com/en/hate-speech-a-faq/a-19103744) puts it like this:"
		"*\"Because there is no one definition of hate speech, it is sometimes difficult to judge what is just an offensive comment and what is hate speech."
		" But basically a nasty comment about an individual isn't hate speech, unless it targets that person as a member of a particular group.\"*"
	)

	dist_df = _get_label_distribution(splits)

	pivot = (
		dist_df.groupby(["Split", "Label"])
		.size()
		.reset_index(name="Count")
		.pivot(index="Label", columns="Split", values="Count")
		.reindex(label_names)
		[["Train", "Validation", "Test"]]
	)

	col_chart, col_table = st.columns(2)

	with col_chart:
		fig, ax = plt.subplots(figsize=(6, 3.5))
		overall = dist_df["Label"].value_counts().reindex(label_names)
		colors = ["#e74c3c", "#2ecc71", "#f39c12"]
		bars = ax.bar(overall.index, overall.values, color=colors)
		for bar, val in zip(bars, overall.values):
			pct = val / overall.sum() * 100
			ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
					f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
		ax.set_ylabel("Number of posts")
		ax.set_title("Overall label distribution")
		ax.set_ylim(0, overall.max() * 1.2)
		fig.tight_layout()
		st.pyplot(fig, use_container_width=False)

	with col_table:
		st.dataframe(pivot, width="stretch")
		max_class = dist_df["Label"].value_counts().max()
		min_class = dist_df["Label"].value_counts().min()
		st.metric("Imbalance ratio", f"{max_class / min_class:.2f}")
		with st.expander("What does this measure mean?"):
			st.markdown("For a dataset with n classes, the imbalance ratio is the count of the largest class (here: Normal) divided by "
						"the count of the smallest class (Offensive). A ratio of 1.0 means perfectly balanced classes, while values "
						"higher than that indicate that some classes are overrepresented compared to others, "
						"which could lead to bias of the model toward the majority class.")


def post_explorer(splits: dict) -> tuple[dict, list[str], str, int]:
	st.subheader("Post Explorer")
	st.write("We are now exploring a single post from the dataset.")
	selected_splits = st.multiselect(
		"Filter by dataset split",
		list(splits.keys()),
		default=list(splits.keys()),
	)
	st.info("To evaluate the classifier's real performance, select only the Test set — the model has never seen these samples during training.")

	if not selected_splits:
		st.warning("Select at least one split.")
		st.stop()

	# Combine selected splits into one list of (split_name, index) pairs
	combined = []
	for split_name in selected_splits:
		ds = splits[split_name]
		for i in range(len(ds)):
			combined.append((split_name, i))

	all_targets = set()
	for split_name, i in combined:
		for annotator_targets in splits[split_name][i]["annotators"]["target"]:
			all_targets.update(annotator_targets)

	selected_targets = st.multiselect("Filter by target group", sorted(all_targets))

	if selected_targets:
		selected_set = set(selected_targets)
		filtered = [
			(s, i) for s, i in combined
			if any(set(targets) & selected_set for targets in splits[s][i]["annotators"]["target"])
		]
	else:
		filtered = combined

	st.write(f"Showing {len(filtered)} posts")

	if not filtered:
		st.warning("No posts match the selected filters.")
		st.stop()

	slider_idx = st.slider("Select a post by adjusting the slider or press the button for a random post", 0, len(filtered) - 1,
						   st.session_state.get("random_idx", 0))

	if st.button("Random"):
		st.session_state["random_idx"] = random.randint(0, len(filtered) - 1)
		st.rerun()

	split_name, post_idx = filtered[slider_idx]
	example = splits[split_name][post_idx]

	st.markdown("#### Post Tokens")
	tokens = example["post_tokens"]
	text = get_post_as_str(tokens)
	st.write(text)

	st.markdown("#### Majority Label")
	st.info("The most common label assigned by annotators is used as the ground truth.")
	ground_truth_id = get_majority_label(example["annotators"]["label"])
	st.markdown(render_label_badge(LABELS[ground_truth_id]), unsafe_allow_html=True)

	st.markdown("#### Highlighted Rationales")
	st.info("Uniquely about this dataset is that besides the annotated labels, it includes token-level rationales "
			"- words that the annotators marked as reasons for the given label.")

	annotator_ids = example["annotators"]["annotator_id"]
	annotator_choice = st.radio(
		"Choose annotator to see rationale:",
		[f"Annotator ID {aid}" for aid in annotator_ids],
		horizontal=True,
		key=f"annotator_{split_name}_{post_idx}",
	)

	selected_id = int(annotator_choice.split()[-1])
	annotator_index = annotator_ids.index(selected_id)

	# Not all annotators have rationales
	if annotator_index < len(example["rationales"]):
		rationale = example["rationales"][annotator_index]
		st.markdown(render_rationale(tokens, rationale), unsafe_allow_html=True)
	else:
		st.info("No rationale available for this annotator.")

	return example, tokens, text, ground_truth_id


def _plot_token_heatmap(values: np.ndarray, tokens: list[str], cmap: str, title: str) -> Figure:
	fig, ax = plt.subplots(figsize=(10, 0.6))
	ax.imshow([values], cmap=cmap, aspect="auto")
	ax.set_yticks([])
	ax.set_xticks(range(len(tokens)))
	ax.set_xticklabels(tokens, rotation=45, ha="right")
	ax.set_title(title)
	return fig


def classifier(text: str, ground_truth_id: int) -> None:
	st.subheader("Hate Speech Classifier")
	st.write(f"Lets see how a classifier trained on the pretrained {MODEL_NAME} performs on the selected post.")

	predicted_label_id = predict_label(text)

	col1, col2, col3 = st.columns(3)
	with col1:
		st.write("Ground truth:")
		st.markdown(render_label_badge(LABELS[ground_truth_id]), unsafe_allow_html=True)
	with col2:
		st.write("Predicted:")
		st.markdown(render_label_badge(LABELS[predicted_label_id]), unsafe_allow_html=True)
	with col3:
		if predicted_label_id != ground_truth_id:
			st.error("Too bad! This post is incorrectly classified.")
		else:
			st.success("Great! This post is correctly classified.")

	st.markdown("#### Try it yourself!")
	custom_text = st.text_area("Enter a mean post 😈")

	if st.button("Predict"):
		custom_label_id = predict_label(custom_text)
		st.markdown(
			f"Predicted: {render_label_badge(LABELS[custom_label_id])}",
			unsafe_allow_html=True,
		)


def explanations(text: str, tokens: list[str], example: dict, ground_truth_id: int) -> tuple[int, list[str]]:
	st.subheader("Explanation Visualization")
	st.write("Back to the selected post from HateXplain. Let's see how the model explains its own prediction.")

	method_name = st.selectbox("Explanation method", list(EXPLANATION_METHODS.keys()))
	method = EXPLANATION_METHODS[method_name]
	importance = method(text)

	# Importance may be shorter than tokens due to truncation
	display_len = min(len(tokens), len(importance))
	importance = importance[:display_len]
	display_tokens = tokens[:display_len]

	fig = _plot_token_heatmap(importance, display_tokens, "Reds", f"Token-level Importance ({method_name})")
	st.pyplot(fig, use_container_width=False)

	with st.expander("How do these explanation methods work?"):
		st.markdown(
			"- **Attention**: Used as baseline. The attention weights are extracted from the last layer of the model. "
			"This shows how much the [CLS] token attends to each token from the post from the input, "
			" averaged across all attention heads. "
			"Tokens that receive more attention are considered more important for the prediction.\n"
			"- **Integrated Gradients**: A gradient-based attribution method that computes the importance of each token "
			"by accumulating gradients along a path from a neutral baseline (zero embedding) to the actual input. "
			"This captures how much each token contributes to the predicted class."
		)

	return display_len, display_tokens


def evaluation(example: dict, text: str, ground_truth_id: int, display_len: int, display_tokens: list[str]) -> None:
	st.markdown("#### Model Evaluation on Test Set")
	st.write("**Performance based metrics**")
	st.info("These metrics are computed on the full test set (1924 samples) that the model never saw during training. ")

	with st.expander("What do these metrics mean?"):
		st.markdown(
			"- **Accuracy**: Percentage of posts classified correctly overall.\n"
			"- **F1 Score (macro)**: Harmonic mean of precision and recall, averaged equally across all classes. "
			"Unlike accuracy, it accounts for class imbalance.\n"
			"- **Precision**: Of all posts predicted as a given class, how many actually belong to that class.\n"
			"- **Recall**: Of all posts that actually belong to a given class, how many were correctly identified.\n"
			"- **Confusion Matrix**: Shows how predictions are distributed across classes. "
			"Diagonal entries are correct predictions; off-diagonal entries are misclassifications."
		)

	eval_results = load_evaluation_results()
	if eval_results is None:
		st.warning("Evaluation results not found. Run `python -m hate_speech_xai.scripts.run_evaluation` first.")
		st.stop()

	y_true, y_pred = eval_results
	acc, f1_macro, report = get_classification_report(y_true, y_pred)

	col1, col2 = st.columns(2)
	with col1:
		st.metric("Accuracy", f"{acc:.2%}")
	with col2:
		st.metric("F1 Score (macro)", f"{f1_macro:.2%}")

	st.caption("Compare to Matthews et al. (2021):"
			"Accuracy: 69.8%, F1 Score (macro): 68.7%,")

	metrics_col1, metrics_col2 = st.columns(2)

	with metrics_col1:
		st.write("**Per-class metrics**")
		rows = []
		for label in label_names:
			rows.append({
				"Label": label,
				"Precision": f"{report[label]['precision']:.2%}",
				"Recall": f"{report[label]['recall']:.2%}",
				"F1": f"{report[label]['f1-score']:.2%}",
				"Support": int(report[label]['support']),
			})
		st.table(rows)
		with st.expander("Interpretation"):
			st.markdown(
				"The model achieves comparable performance to Mathew et al. (2021)."
				"We can see that the **Offensive** class is the hardest to classify with almost half of all offensive posts misclassified."
				"**Hate speech** is classified more reliably, with the majority of errors being confusion with the Offensive class."
				"**Normal** posts are predicted best overall, although a notable share is misclassified "
				"as Offensive. This could be doue to sarcastic or ambiguous language. Deciding for a class in cases of doubt"
				" when only judging a post based on the text without additional context like a persons gestures or tone of voice.\n\n"
				"Future work could explore how model performance and quality of explanations change when there are only two "
				"classes: toxic and normal. This could help with the problem that it's hard (even for humans) to disambiguate "
				"between hate speech and offensive speech."
			)

	with metrics_col2:
		st.write("**Confusion Matrix**")
		fig = plot_confusion_matrix(y_true, y_pred)
		st.pyplot(fig, use_container_width=False)
		st.caption(
			"Problem with the Offensive class: 25.7% of offensive posts are confused as Normal and 21.7% as Hate speech. "
			"Similarly, 22.4% of Normal posts are incorrectly labeled Offensive."
		)

	st.divider()

	st.write("**Explainability metrics**")
	st.info("How well do the explanation methods match the ground truth rationales from the dataset?")

	with st.expander("What do these metrics mean?"):
		st.markdown(
			"- **Top-k Overlap**: Of the top-k tokens the method considers most important, "
			"how many match the ground truth rationale tokens? (k = number of rationale tokens)"
		)

	xai_results = load_xai_evaluation_results()
	if xai_results is None:
		st.warning("XAI evaluation results not found. Run `make evaluate-xai` first.")
		return

	xai_rows = []
	for r in xai_results:
		xai_rows.append({
			"Method": r["method"],
			"Top-k Overlap": f"{r['top_k_overlap']:.2%}" if r["top_k_overlap"] else "—",
			"Samples": r["n_samples"],
		})
	st.table(xai_rows)
	st.caption(
		"This evaluation is only based on 1,135 out of 1,924 test samples. "
		"Excluded are posts labeled as **Normal** because their ground truth rationale is uniform "
		", which makes a top-k overlap comparison meaningless."
	)

	st.write("**Direct comparison of XAI methods evaluation on the selected post from above**")

	gt_rationale_eval = aggregate_rationales(example["rationales"], ground_truth_id)
	gt_display_eval = gt_rationale_eval[:display_len]

	fig_gt = _plot_token_heatmap(gt_display_eval, display_tokens, "Oranges", "Ground Truth Rationale")
	st.pyplot(fig_gt, use_container_width=False)

	for r in xai_results:
		method_name_eval = r["method"]
		if method_name_eval not in EXPLANATION_METHODS:
			continue
		method_fn = EXPLANATION_METHODS[method_name_eval]
		imp = method_fn(text)
		imp_display = imp[:display_len]

		title = method_name_eval
		top_k_eval_result = top_k_overlap(imp, gt_rationale_eval)
		if top_k_eval_result is not None:
			title = f"{method_name_eval} — Top-k Overlap: {top_k_eval_result:.2%}"

		fig = _plot_token_heatmap(imp_display, display_tokens, "Reds", title)
		st.pyplot(fig, use_container_width=False)
