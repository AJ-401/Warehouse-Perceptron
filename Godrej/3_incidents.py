"""
FIELD INTELLIGENCE - PAGE 3: INCIDENT INVESTIGATION & NEAR-MISS REPLAY
Run with:  streamlit run 3_incidents.py
Requires incidents.html to sit in the SAME folder as this script.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="Field Intelligence // Incidents",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu, header, footer {visibility: hidden;}
        div.block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
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

HTML_FILE = Path(__file__).parent / "incidents.html"
html_code = HTML_FILE.read_text(encoding="utf-8")

components.html(html_code, height=1000, scrolling=True)
