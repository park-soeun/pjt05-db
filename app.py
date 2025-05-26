import streamlit as st
import pandas as pd
import plotly.express as px

# --- 페이지 설정 ---
st.set_page_config(
    page_title="주조 공정 모니터링 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 사이드바 ---
st.sidebar.title("주조 공정 대시보드")
st.sidebar.markdown("""
- 프로젝트명: 주조 공정 센서 데이터 기반 공정 모니터링 대시보드 개발
- 목적: 실시간 품질 예측 및 이상 탐지 시스템 구축
""")

menu = st.sidebar.radio("탭 선택", [
    "📊 공정 이상 탐지",
    "🧠 불량 예측 모델",
    "🔍 원인 분석 (XAI)",
    "📈 실시간 모니터링"
])

# --- 공정 이상 탐지 탭 ---
if menu == "📊 공정 이상 탐지":
    st.title("공정 이상 탐지")
    st.markdown("Z-score, PCA, DBSCAN 등 비지도 학습 기반 이상 탐지 시각화")

    st.subheader("1. 공정 변수 분포 및 이상 탐지 결과")
    st.info("💡 여기에 Z-score나 DBSCAN 결과 그래프 삽입 예정")

    # placeholder for future plot
    st.empty()

# --- 불량 예측 모델 탭 ---
elif menu == "🧠 불량 예측 모델":
    st.title("불량 예측 모델")
    st.markdown("사용자 입력 기반 실시간 불량 예측 및 시각화")

    with st.form("predict_form"):
        st.subheader("1. 공정 변수 입력")
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.number_input("주형 온도", value=500)
        with col2:
            pressure = st.number_input("주입 압력", value=100)
        with col3:
            speed = st.number_input("주입 속도", value=20)

        submitted = st.form_submit_button("불량 예측")

        if submitted:
            st.success("예측 결과: 정상 (정확도 92%)")  # placeholder 결과

# --- 원인 분석 탭 ---
elif menu == "🔍 원인 분석 (XAI)":
    st.title("불량 원인 분석 및 변수 영향도")
    st.markdown("SHAP, Permutation Importance 등을 통한 설명 가능한 AI 분석")

    st.subheader("1. 변수 영향도 시각화")
    st.info("💡 주요 변수별 SHAP/PI 그래프 추가 예정")

    st.subheader("2. 특정 샘플 예측 근거 시각화")
    sample_id = st.selectbox("샘플 선택", options=["샘플 1", "샘플 2", "샘플 3"])
    st.info(f"💡 {sample_id}의 예측 근거 분석 예정")

# --- 실시간 모니터링 탭 ---
elif menu == "📈 실시간 모니터링":
    st.title("실시간 공정 모니터링")
    st.markdown("공정 변수 트렌드, 불량률, 이상 감지 결과를 실시간 표시")

    st.subheader("1. 공정 변수 시계열 그래프")
    st.info("💡 실시간 시계열 데이터 흐름 표시 예정")

    st.subheader("2. 공정 품질 상태 현황판")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("현재 불량률", "3.1%", "-0.4%")
    with col2:
        st.metric("이상 탐지 건수", "2건", "+1")

    st.subheader("3. 주요 변수 변화 추이")
    st.info("💡 중요 변수별 트렌드 라인 차트 예정")