"""
TextInsight app entrypoint
"""
import streamlit as st

# Importar configuración de TensorFlow
from src import config_tensorflow

from src.gui import run_app

st.set_page_config(
    page_title='TextInsight',
    page_icon='logo.png',
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': None,
        'About': "# Esta es una aplicación de ejemplo"
    }
)

if __name__ == "__main__":
    run_app()
