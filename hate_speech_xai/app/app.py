import streamlit as st

from hate_speech_xai.app.design import give_credit_to_photographer2, add_styling
from hate_speech_xai.config import MODEL_NAME, LABELS
from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset
from hate_speech_xai.src.data.preprocessing import get_majority_label
from hate_speech_xai.src.models.predict import predict

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

st.write("Label meaning: 0=hatespeech, 1=normal, 2=offensive")

st.subheader("Highlighted rationales")
rationale = example["rationales"][0]

highlighted = []
for token, r in zip(tokens, rationale):
    if r == 1:
        highlighted.append(f"**{token}**")
    else:
        highlighted.append(token)

st.write(" ".join(highlighted))


##############################################################
st.subheader("Hate Speech Classifier")
st.write("Trained model: ", MODEL_NAME)
text = " ".join(example["post_tokens"])
predicted_label = predict(text)
predicted_label = LABELS[predicted_label]
st.write("Predicted Label")
st.write(predicted_label)

##############################################################
st.subheader("Try it yourself!")
custom_text = st.text_area("Enter a post")

if st.button("Predict"):
    predicted_label = predict(custom_text)
    predicted_label = LABELS[predicted_label]

    #st.write("Prediction:", label)
    #st.write("Probabilities:", probs)

##############################################################
st.subheader("Dummy Explanation (for demo purposes)")
import numpy as np
import matplotlib.pyplot as plt

# Dummy prediction
labels = ["normal", "offensive", "hate"]
pred_label = np.random.choice(labels)
st.subheader("Predicted Label")
st.write(pred_label)

# Token-level importance (random placeholder)
importance = np.random.rand(len(tokens))

# Normalize for visualization
importance = importance / importance.max()

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 0.6))
ax.imshow([importance], cmap="Reds", aspect="auto")
ax.set_yticks([])
ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha='right')
ax.set_title("Token-level Importance (Random for Demo)")

st.pyplot(fig)

# Show ground truth rationales
if st.checkbox("Show ground truth rationales"):
    # Union across annotators
    import numpy as np
    rationales = np.array(example['rationales'])
    rationale_mask = np.max(rationales, axis=0)
    st.write("Rationale mask (1=important token):")
    st.write(rationale_mask)

give_credit_to_photographer2()
