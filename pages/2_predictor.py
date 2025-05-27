import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="호출 2: 불량 예측", layout="wide")
st.title("🤖 불량 예측 시리워")

# --- 모델 로딩 ---
@st.cache_resource

def load_model():
    try:
        model = joblib.load("../model/random_forest_model.pkl")
        return model
    except FileNotFoundError:
        st.error("\U0001F6A8 분류 모델(random_forest_model.pkl)을 찾을 수 없습니다. '..model/' 폴더에 저장해 주세요.")
        return None

model = load_model()

# --- 입력 받을 변수 정의 ---
st.sidebar.subheader("형성자 입력")

# 예시 입력값 (필요에 따라 수정)
input_fields = {
    'molten_temp': st.sidebar.number_input('molten_temp', 500.0, 800.0, 700.0),
    'cast_pressure': st.sidebar.number_input('cast_pressure', 100.0, 400.0, 300.0),
    'physical_strength': st.sidebar.number_input('physical_strength', 0.0, 1500.0, 700.0),
    'biscuit_thickness': st.sidebar.number_input('biscuit_thickness', 0.0, 100.0, 50.0),
    'Coolant_temperature': st.sidebar.number_input('Coolant_temperature', 0.0, 100.0, 30.0),
    'EMS_operation_time': st.sidebar.number_input('EMS_operation_time', 0, 25, 10)
}

input_df = pd.DataFrame([input_fields])

if model:
    st.subheader(":mag: 예측 결과")
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # 불량 확률

    if prediction == 1:
        st.error(f"\U0001F6A8 불량 가능성 있음! (확률: {prob:.2%})")
    else:
        st.success(f"\u2705 양품으로 예측됨 (불량 확률: {prob:.2%})")

    with st.expander("테스트 입력 데이터 보기"):
        st.write(input_df)
else:
    st.info("모델이 로드되지 않았습니다. 좌측 사이드바에서 입력을 해보세요.")
