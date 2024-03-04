# main.py
import streamlit as st

st.set_page_config(page_title='TextInsight', page_icon='TextInsight_logo_2.jpg')

from src.gui import run_app

if __name__ == "__main__":
    run_app()
