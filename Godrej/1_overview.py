"""
FIELD INTELLIGENCE - PAGE 1: OVERVIEW (LANDING / DASHBOARD)
Run with:  streamlit run 1_overview.py
Requires overview.html to sit in the SAME folder as this script.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="Field Intelligence // Overview",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu, header, footer {visibility: hidden;}
        div.block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
        /* Stretch the embedded iframe to fill the actual browser window
           (100vw x 100vh). The page's own min-h-screen layout then sizes
           itself to your real screen, exactly like opening the HTML
           directly. scrolling stays enabled below so that if the page's
           content is taller than one screen, it scrolls internally
           instead of leaving dead space or getting clipped. */
        iframe {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

HTML_FILE = Path(__file__).parent / "overview.html"
html_code = HTML_FILE.read_text(encoding="utf-8")

components.html(html_code, height=1000, scrolling=True)
