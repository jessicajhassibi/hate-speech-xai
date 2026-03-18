import base64
from pathlib import Path

import streamlit as st

from hate_speech_xai.config import DATA_DIR


def add_styling_photo(background_image_name: str):
    image_path = DATA_DIR / background_image_name

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
        <style>
        header {{
            background: transparent !important;
        }}

        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), url("data:image/jpg;base64,{encoded}");
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

        /* Selectbox and multiselect input */
        [data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
        }}
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {{
            color: white !important;
        }}
        [data-baseweb="select"] svg {{
            fill: white !important;
        }}

        /* Dropdown options */
        [data-baseweb="popover"] span,
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] div {{
            color: black !important;
        }}

        /* Multiselect selected tags */
        [data-baseweb="tag"] {{
            background-color: rgba(255, 255, 255, 0.9) !important;
        }}
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] svg {{
            color: black !important;
            fill: black !important;
        }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def add_styling_professional():
    css = """
        <style>
        header {
            background: transparent !important;
        }

        .stApp {
            background-color: #f8f9fa;
        }

        .block-container {
            max-width: 900px;
            padding: 2rem 2.5rem;
        }

        /* Section containers */
        .stSubheader {
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
        }

        /* Text elements */
        h1 {
            color: #1a1a2e !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
        }

        h2, h3, h4, h5, h6 {
            color: #2d2d44 !important;
            font-weight: 600 !important;
        }

        p, div, label, span, li {
            color: #333333 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #1a1a2e;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.2s;
        }

        .stButton > button:hover {
            background-color: #2d2d44;
        }

        /* Text input and text area */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: white !important;
            color: #333333 !important;
            border: 1px solid #d0d0d0 !important;
            border-radius: 8px !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #1a1a2e !important;
            box-shadow: 0 0 0 1px #1a1a2e !important;
        }

        /* Slider */
        .stSlider [data-baseweb="slider"] {
            background: #e0e0e0;
        }

        /* Radio buttons */
        .stRadio label {
            color: #333333 !important;
        }

        /* Multiselect tags */
        [data-baseweb="tag"] {
            background-color: #1a1a2e !important;
            border-radius: 6px !important;
        }
        [data-baseweb="tag"] span {
            color: white !important;
        }
        [data-baseweb="tag"] svg {
            fill: white !important;
        }

        /* Info and warning boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Selectbox and multiselect input */
        [data-baseweb="select"] > div {
            background-color: white !important;
            border: 1px solid #d0d0d0 !important;
            border-radius: 8px !important;
        }
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {
            color: #333333 !important;
        }
        [data-baseweb="select"] svg {
            fill: #333333 !important;
        }

        /* Dropdown options */
        [data-baseweb="popover"] span,
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] div {
            color: #333333 !important;
        }
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