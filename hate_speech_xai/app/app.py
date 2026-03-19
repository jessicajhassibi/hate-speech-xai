import random
import streamlit as st
from matplotlib import pyplot as plt

from hate_speech_xai.app.styling import (
	THEMES, apply_theme, render_label_badge, render_rationale,
	render_speech_bubble, render_photo_credit,
)
from hate_speech_xai.config import MODEL_NAME, LABELS
from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset
from hate_speech_xai.src.data.preprocessing import get_majority_label
from hate_speech_xai.src.models.predict import predict_label
from hate_speech_xai.src.models.explain import EXPLANATION_METHODS

st.set_page_config(page_title="Hate Speech XAI", layout="wide")

theme = st.sidebar.selectbox("Theme", THEMES, label_visibility="collapsed")
apply_theme(theme)

st.title("Hate Speech XAI")

train_ds, val_ds, test_ds = load_hatexplain_dataset()
splits = {"Train": train_ds, "Validation": val_ds, "Test": test_ds}

st.subheader("HateXplain Dataset Explorer")

selected_splits = st.multiselect(
	"Filter by dataset split",
	list(splits.keys()),
	default=list(splits.keys()),
	help="For exploring the dataset To evaluate the classifier's real performance, you should select only the Test set, becausex "
		 "the model has never seen these samples during training.",
)

if not selected_splits:
	st.warning("Select at least one split.")
	st.stop()

# Combine selected splits into one list of (split_name, index) pairs
combined = []
for split_name in selected_splits:
	ds = splits[split_name]
	for i in range(len(ds)):
		combined.append((split_name, i))

# Collect all target categories
all_targets = set()
for split_name, i in combined:
	for annotator_targets in splits[split_name][i]["annotators"]["target"]:
		all_targets.update(annotator_targets)

selected_targets = st.multiselect(
	"Filter by target group",
	sorted(all_targets),
)

if selected_targets:
	selected_set = set(selected_targets)
	filtered = [
		(s, i) for s, i in combined
		if any(
			set(targets) & selected_set
			for targets in splits[s][i]["annotators"]["target"]
		)
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

st.subheader("Post Tokens")
tokens = example["post_tokens"]
text = " ".join(tokens)
st.markdown(render_speech_bubble(text, dark_bg=(theme == "Dark")), unsafe_allow_html=True)

st.subheader("Ground Truth (Majority Label)", help="The most common label assigned by annotators is used as the "
											   "ground truth for this post. There are 3 possible labels: "
											   "hate speech, offensive, and normal")
ground_truth_id = get_majority_label(example["annotators"]["label"])
st.markdown(render_label_badge(LABELS[ground_truth_id]), unsafe_allow_html=True)

st.subheader("Highlighted Rationales")

annotator_ids = example["annotators"]["annotator_id"]
annotator_choice = st.radio(
	"Choose annotator",
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

st.subheader("Hate Speech Classifier")
st.write("Trained model: ", MODEL_NAME)

predicted_label_id = predict_label(text)

col1, col2 = st.columns(2)
with col1:
	st.write("Ground truth:")
	st.markdown(render_label_badge(LABELS[ground_truth_id]), unsafe_allow_html=True)
with col2:
	st.write("Predicted:")
	st.markdown(render_label_badge(LABELS[predicted_label_id]), unsafe_allow_html=True)

st.subheader("Try it yourself!")
custom_text = st.text_area("Enter a mean post 😈")

if st.button("Predict"):
	custom_label_id = predict_label(custom_text)
	st.markdown(
		f"Predicted: {render_label_badge(LABELS[custom_label_id])}",
		unsafe_allow_html=True,
	)

st.subheader("Explanation Visualization")

method_name = st.selectbox("Explanation method", list(EXPLANATION_METHODS.keys()))
explain_fn = EXPLANATION_METHODS[method_name]
importance = explain_fn(text)

# Importance may be shorter than tokens due to truncation
display_len = min(len(tokens), len(importance))
importance = importance[:display_len]
display_tokens = tokens[:display_len]

fig, ax = plt.subplots(figsize=(10, 0.6))
ax.imshow([importance], cmap="Reds", aspect="auto")
ax.set_yticks([])
ax.set_xticks(range(len(display_tokens)))
ax.set_xticklabels(display_tokens, rotation=45, ha="right")
ax.set_title(f"Token-level Importance ({method_name})")
st.pyplot(fig)

if theme == "Dark":
	render_photo_credit()