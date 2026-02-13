import streamlit as st
from hate_speech_xai.src.load_hatexplain import load_hatexplain_dataset


st.set_page_config(page_title="Hate Speech XAI", layout="wide")
st.title("Hate Speech XAI")

train_ds, val_ds, test_ds = load_hatexplain_dataset()

###################################################

st.subheader("HateXplain Sample Posts")
for i, example in enumerate(train_ds.select(range(5))):
    st.subheader(f"Post {i+1}")
    st.write(" ".join(example['post_tokens']))
    st.write("Label:", example['annotators']['label'])
    st.write("Rationale:", example['rationales'])
    st.write("Post Tokens:", example['post_tokens'])
