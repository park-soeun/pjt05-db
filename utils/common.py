import streamlit as st
import os

# styles.css 읽어서 삽입
def inject_css():
    with open(os.path.join("www", "styles.css")) as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)