import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="호출 3: 변수 영향도 (XAI)", layout="wide")
st.title("\U0001f50d 예측 결과 및 변수 영향도 분석")

# --- 모델 및 샘플 로딩 ---
def load_model():
    try:
        return joblib.load("../model/classifier.pkl")
    except:
        st.error("classifier.pkl 파일을 \\\"../model/\\\" 폴더에 저장해주세요.")
        return None

@st.cache_data

def load_data():
    try:
        return pd.read_csv("../data/train.csv")
    except:
        st.error("classifier.pkl 파일을 \\\"../model/\\\" 폴더에 저장해주세요.")
        return None

model = load_model()
data = load_data()

if model and data is not None:
    # --- Feature 선택 ---
    feature_names = [col for col in data.columns if col not in ['id', 'passorfail', 'datetime']]
    sample_index = st.slider("샘플 선택 (index)", 0, len(data) - 1, 0)
    X = data[feature_names]
    y = data['passorfail']

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # --- Force Plot for selected sample ---
    st.subheader(f"선택한 샘플 (index={sample_index})의 SHAP 예측 근거")
    shap.initjs()
    st_shap = st.components.v1.html(shap.plots.force(explainer.expected_value, shap_values[sample_index].values, X.iloc[sample_index], matplotlib=False), height=300)

    # --- 전체 feature 영향도 ---
    st.subheader("전체 변수 중요도 (SHAP Summary Plot)")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

else:
    st.info("모델과 학습 데이터를 먼저 확인해주세요. \\n필요한 파일: '../model/classifier.pkl', '..data/train.csv'")