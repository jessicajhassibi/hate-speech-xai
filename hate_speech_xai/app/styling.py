import base64

import streamlit as st

from hate_speech_xai.config import APP_DATA_DIR

THEMES = ["Professional", "Dark"]

# Use the same colors for the labels throughout the app
LABEL_COLORS = {
    "Hate speech": "#d32f2f",
    "Normal": "#388e3c",
    "Offensive": "#f57c00",
}


def render_label_badge(label_name: str) -> str:
    color = LABEL_COLORS[label_name]
    html = f'<span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 600;">{label_name}</span>'
    return html



def render_rationale(tokens: list[str], rationale: list[int]) -> str:
    parts = []
    for token, r in zip(tokens, rationale):
        if r == 1:
            parts.append(f'<span style="background-color: #f0a500; color: white; padding: 2px 4px; border-radius: 4px;">{token}</span>')
        else:
            parts.append(token)
    return " ".join(parts)


def render_photo_credit() -> None:
    """For the "dark" theme we use a free photo from Unsplash. We need to give credit to the photographer."""

    st.markdown(
        '<div style="position: fixed; bottom: 10px; right: 15px; font-size: 12px; color: white; opacity: 0.7;">'
        'Photo by <a href="https://unsplash.com/@adrienolichon" target="_blank" style="color: white;">Adrien Olichon</a>'
        ' on <a href="https://unsplash.com" target="_blank" style="color: white;">Unsplash</a></div>',
        unsafe_allow_html=True,
    )


def apply_theme(theme: str) -> None:
    if theme == "Dark":
        _apply_dark_theme()
    else:
        _apply_professional_theme()


def _apply_dark_theme() -> None:
    """Basically, it applies a dark background picture (can be switched) and changes all the fonts in the elements to white so we have an appropriate contrast.
    Also the box backgrounds needed to be transparent and some other details."""

    image_path = APP_DATA_DIR / "background.jpg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    header {{
        background: transparent !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        background: transparent !important;
    }}

    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
                          url("data:image/jpg;base64,{image_data}");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        max-width: 1400px;
        padding: 2rem 2.5rem;
    }}

    h1, h2, h3, h4, h5, h6, p, div, label, span, li {{
        color: white !important;
    }}

    .stButton > button {{
        background-color: rgba(255, 255, 255, 0.2);
        color: white !important;
        border: 1px solid white;
    }}
    .stButton > button span {{
        color: white !important;
    }}
    .stDownloadButton > button {{
        background-color: rgba(255, 255, 255, 0.2);
        color: white !important;
        border: 1px solid white;
    }}
    .stDownloadButton > button span {{
        color: white !important;
    }}

    textarea, input {{
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: black !important;
        -webkit-text-fill-color: black !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
    }}

    [data-baseweb="select"] > div {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {{
        color: white !important;
    }}
    [data-baseweb="select"] svg {{
        fill: white !important;
    }}

    [data-baseweb="popover"] li span,
    [data-baseweb="popover"] li div,
    [data-baseweb="popover"] ul span {{
        color: black !important;
    }}
    
    /* Info boxes: width should match the content width*/
    .stAlert {{
        width: fit-content !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _apply_professional_theme() -> None:
    """Applies custom colors and other details for the professional theme."""

    css = """
    <style>
    header {
        background: transparent !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    .stApp {
        background: linear-gradient(160deg, #f0f2f5 0%, #e8ecf1 50%, #f0f2f5 100%);
    }
    .block-container {
        max-width: 1400px;
        padding: 2rem 2.5rem;
    }

    h1 {
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }
    h2, h3, h4, h5, h6 {
        color: #2d2d44 !important;
        font-weight: 600 !important;
    }
    p, div, label, span, li {
        color: #333 !important;
    }

    textarea, input[type="text"] {
        background-color: white !important;
        color: #333 !important;
    }

    /* Info boxes: fit content width */
    .stAlert {
        width: fit-content !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)