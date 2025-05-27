import streamlit as st
from views.monitoring import create_realtime_monitoring
from utils.common import inject_css 

inject_css()
create_realtime_monitoring()
