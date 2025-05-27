import streamlit as st
from utils.loader import load_models
from views.quality import create_quality_prediction_interface
from utils.common import inject_css 

inject_css()
models = load_models()
create_quality_prediction_interface(models)