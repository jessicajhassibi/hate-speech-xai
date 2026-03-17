import base64
from pathlib import Path

import streamlit as st

from hate_speech_xai.config import DATA_DIR

def add_styling(background_image_name: str):
    image_path = DATA_DIR / background_image_name

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
        <style>
        header {{
            background: transparent !important;
        }}

        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            padding: 2rem;
            color: white;
        }}

        /* Slider and input labels */
        .stSlider label, .stTextArea label, .stButton label {{
            color: white !important;
        }}

        /* Text elements */
        h1, h2, h3, h4, h5, h6, p, div, label, span {{
            color: white !important;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: rgba(255, 255, 255, 0.2);
            color: white !important;
            border: 1px solid white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s;
        }}

        .stButton > button:hover {{
            background-color: rgba(255, 255, 255, 0.3);
            border: 1px solid white;
        }}

        /* Text input and text area */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            color: black !important;
            border: 1px solid rgba(0, 0, 0, 0.3) !important;
        }}

        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {{
            color: rgba(0, 0, 0, 0.4) !important;
        }}

        /* Slider track */
        .stSlider [data-baseweb="slider"] {{
            background: rgba(255, 255, 255, 0.2);
        }}

        /* Checkbox */
        .stCheckbox label {{
            color: white !important;
        }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def give_credit_to_photographer1():
    st.markdown(
        """
        <div style="position: fixed; bottom: 10px; right: 15px; 
                    font-size: 12px; color: white; opacity: 0.7;">
            Photo by <a href="https://unsplash.com/@jontyson" target="_blank" style="color: white;">
            Jon Tyson </a> on <a href="https://unsplash.com" target="_blank" style="color: white;">
            Unsplash</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def give_credit_to_photographer2():
    st.markdown(
        """
        <div style="position: fixed; bottom: 10px; right: 15px; 
                    font-size: 12px; color: white; opacity: 0.7;">
            Photo by <a href="https://unsplash.com/@anafsnt" target="_blank" style="color: white;">
            Ana Flávia </a> on <a href="https://unsplash.com" target="_blank" style="color: white;">
            Unsplash</a>
        </div>
        """,
        unsafe_allow_html=True
    )