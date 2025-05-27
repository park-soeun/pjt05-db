import streamlit as st
from views.advanced import create_advanced_analytics
from utils.common import inject_css 

inject_css()
create_advanced_analytics()
