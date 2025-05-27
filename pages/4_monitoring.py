import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

st.set_page_config(page_title="호출 4: 실시간 공정 모니터링", layout="wide")
st.title("\U0001f4c8 실시간 공정 모니터링")

# --- 데이터 불러오기 ---
@st.cache_data

def load_data():
    try:
        return pd.read_csv("../data/train.csv", parse_dates=["datetime"])
    except:
        st.error()
        return None

df = load_data()

if df is not None:
    st.sidebar.header("보여주기 전체 값")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    num_cols = [col for col in num_cols if col not in ['id', 'passorfail']]

    selected_var = st.sidebar.selectbox("시계열 구현 변수", num_cols)
    interval = st.sidebar.slider("실시간 보기 상실화 간격 (초)", 3, 30, 5)

    # 최신 100개만 보기
    df_latest = df.sort_values("datetime").tail(100)

    st.subheader(f"\U0001f4c5 최신 공정 데이터 트렌드 - {selected_var}")
    fig = px.line(df_latest, x="datetime", y=selected_var, title=f"최근 {selected_var} 시계열")
    st.plotly_chart(fig, use_container_width=True)

    # 불량률 추이
    st.subheader("\U0001f6a8 최근 불량률 변화")
    df['hour'] = df['datetime'].dt.hour
    hour_rate = df.groupby('hour')['passorfail'].mean()
    fig2 = px.bar(x=hour_rate.index, y=hour_rate.values, labels={'x': '시간', 'y': '불량률'}, title='시간대별 평균 불량률')
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"{interval}초마다 새로고침 하세요 (Streamlit에는 자동 갱신 기능 없음 → 수동으로 새로고침하거나 페이지 새로 열기)")
else:
    st.warning()