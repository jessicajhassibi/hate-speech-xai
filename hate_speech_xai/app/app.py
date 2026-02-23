from collections import Counter

import streamlit as st
from hate_speech_xai.src.load_hatexplain import load_hatexplain_dataset


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

st.subheader("Majority Label")
majority_label = Counter(example['annotators']['label']).most_common(1)[0][0]
st.write(majority_label)

st.subheader("Rationales")
st.write("Rationale:", example['rationales'])


