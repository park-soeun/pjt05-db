import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="호출 5: 성능 리포트", layout="wide")
st.title("\U0001f4cb 모델 성능 리포트")

# --- 데이터 로딩 ---
def load_data():
    try:
        df = pd.read_csv("../data/train.csv")
        return df
    except:
        st.error("../data/train.csv 파일을 찾을 수 없습니다.")
        return None

def load_model():
    try:
        return joblib.load("../model/classifier.pkl")
    except:
        st.error("../model/classifier.pkl 파일을 찾을 수 없습니다.")
        return None

df = load_data()
model = load_model()

if df is not None and model:
    st.subheader("배포 전 모델 성능")
    feature_names = [col for col in df.select_dtypes(include='number').columns if col not in ['id', 'passorfail']]
    X = df[feature_names]
    y = df['passorfail']

    y_pred = model.predict(X)

    st.markdown("### ▶️ Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["정상", "불량"], yticklabels=["정상", "불량"])
    plt.xlabel("예측")
    plt.ylabel("실제")
    st.pyplot(fig)

    st.markdown("### ▶️ Classification Report")
    report = classification_report(y, y_pred, target_names=["정상", "불량"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### ▶️ 변수 간 상관관계 (Heatmap)")
    corr = df[feature_names + ['passorfail']].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    st.pyplot(fig2)

    st.markdown("### ▶️ 모델 저장")
    with open("../model/classifier.pkl", "rb") as f:
        st.download_button("파일 다운로드", f, file_name="classifier.pkl")
else:
    st.warning("모델 또는 train.csv 데이터를 로드할 수 없습니다.")