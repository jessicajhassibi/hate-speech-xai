import streamlit as st

from hate_speech_xai.app.styling import THEMES, apply_theme, render_photo_credit
from hate_speech_xai.config import APP_DATA_DIR, REPORT_DIR, REPORT_NAME, REPORT_PATH
from hate_speech_xai.app.sections import (
	dataset_explorer,
	post_explorer,
	classifier,
	explanations,
	evaluation,
)
from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset

st.set_page_config(layout="wide")
st.sidebar.markdown("### Choose Theme")
theme = st.sidebar.selectbox("Theme", THEMES, label_visibility="collapsed")
apply_theme(theme)

st.sidebar.markdown("### Navigation")
st.sidebar.markdown("""
- [Dataset Explorer](#hatexplain-dataset-explorer)
- [Post Explorer](#post-explorer)
- [Hate Speech Classifier](#hate-speech-classifier)
- [Explanation Visualization](#explanation-visualization)
- [Model Evaluation](#model-evaluation-on-test-set)
""")

if REPORT_PATH.exists():
	st.sidebar.markdown("### Report")
	with open(REPORT_PATH, "rb") as f:
		st.sidebar.download_button("Download Report (PDF)", f.read(), file_name=REPORT_NAME, mime="application/pdf")

st.title("Explainable AI for Hate Speech Detection")
st.caption("Practical Project made by Jessica Hassibi, Winter Semester 2025/26 for the Practical Course AI and Security by TU Darmstadt "
		   "and the Fraunhofer SIT of Darmstadt.")
st.caption("This project is based on and builds upon the research of [Mathew et al., 2021](https://doi.org/10.1609/aaai.v35i17.17745)")

train_ds, val_ds, test_ds = load_hatexplain_dataset()
splits = {"Train": train_ds, "Validation": val_ds, "Test": test_ds}

# now we run the code for the different sections of the app, which are defined in sections.py
dataset_explorer(splits)
st.divider()

example, tokens, text, ground_truth_id = post_explorer(splits)
st.divider()

classifier(text, ground_truth_id)
st.divider()

display_len, display_tokens, cached_importance = explanations(text, tokens, example, ground_truth_id)
st.divider()

evaluation(example, text, ground_truth_id, display_len, display_tokens, cached_importance)

if theme == "Dark":
	render_photo_credit()