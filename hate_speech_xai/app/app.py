import streamlit as st
from matplotlib import pyplot as plt

from hate_speech_xai.app.design import give_credit_to_photographer2, add_styling
from hate_speech_xai.config import MODEL_NAME, LABELS
from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset
from hate_speech_xai.src.data.preprocessing import get_majority_label
from hate_speech_xai.src.models.predict import predict_label
from hate_speech_xai.src.models.explain import EXPLANATION_METHODS

add_styling("background_image2.jpg")

st.set_page_config(page_title="Hate Speech XAI", layout="wide")
st.title("Hate Speech XAI")

train_ds, val_ds, test_ds = load_hatexplain_dataset()

###################################################
st.subheader("HateXplain Sample Posts")

post_idx = st.slider("Select a post from the training set", 0, len(train_ds)-1, 0)
example = train_ds[post_idx]

tokens = example['post_tokens']
st.subheader("Post Tokens")
st.write(" ".join(tokens))

st.subheader("Majority Label as ground truth")
st.write(get_majority_label(example["annotators"]["label"]))

st.write(f"Label meaning: 0 = {LABELS[0]}, 1 = {LABELS[1]}, 2 = {LABELS[2]}")

st.subheader("Highlighted rationales")

def render_rationale(tokens, rationale):
	highlighted = []

	for token, r in zip(tokens, rationale):
		if r == 1:
			highlighted.append(
				f'<span style="background-color: #ff4b4b; color: white; padding: 2px 4px; border-radius: 4px;">{token}</span>'
			)
		else:
			highlighted.append(token)

	return " ".join(highlighted)

annotator_ids = example["annotators"]["annotator_id"]

annotator_choice = st.radio(
	"Choose annotator",
	[f"Annotator ID {aid}" for aid in annotator_ids],
	horizontal=True,
	key=f"annotator_{post_idx}"
)

# Extract the selected annotator ID and find its index
selected_id = int(annotator_choice.split()[-1])
annotator_index = annotator_ids.index(selected_id)

# Get the rationale for the selected annotator (not all annotators have rationales)
if annotator_index < len(example["rationales"]):
	rationale = example["rationales"][annotator_index]
	html = render_rationale(tokens, rationale)
	st.markdown(html, unsafe_allow_html=True)
else:
	st.info("No rationale available for this annotator.")


##############################################################
st.subheader("Hate Speech Classifier")
st.write("Trained model: ", MODEL_NAME)
text = " ".join(example["post_tokens"])
predicted_label = predict_label(text)
predicted_label = LABELS[predicted_label]
st.write("Predicted Label")
st.write(predicted_label)

##############################################################
st.subheader("Try it yourself!")
custom_text = st.text_area("Enter a mean post 😈")

if st.button("Predict"):
    predicted_label_for_custom_text = predict_label(custom_text)
    predicted_label_for_custom_text = LABELS[predicted_label_for_custom_text]
    st.write("Predicted Label for your post: ", predicted_label_for_custom_text)

##############################################################
st.subheader("Explanation Visualization")

method_name = st.selectbox("Explanation method", list(EXPLANATION_METHODS.keys()))
explain_fn = EXPLANATION_METHODS[method_name]

importance = explain_fn(text)

# Align lengths (importance may be shorter/longer than tokens due to truncation)
display_len = min(len(tokens), len(importance))
importance = importance[:display_len]
display_tokens = tokens[:display_len]

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 0.6))
ax.imshow([importance], cmap="Reds", aspect="auto")
ax.set_yticks([])
ax.set_xticks(range(len(display_tokens)))
ax.set_xticklabels(display_tokens, rotation=45, ha='right')
ax.set_title(f"Token-level Importance ({method_name})")

st.pyplot(fig)

give_credit_to_photographer2()
