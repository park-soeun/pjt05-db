import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="호출 1: 이상 탐지", layout="wide")
st.title("📊 공정 변수 이상 탐지")

# --- 데이터 불러오기 ---
st.sidebar.header("데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("데이터 업로드 완료")

    # --- 수치형 데이터만 사용 ---
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if 'id' in num_cols:
        num_cols.remove('id')

    st.sidebar.subheader("📌 사용 변수 선택")
    selected_features = st.sidebar.multiselect("이상 탐지에 사용할 변수", num_cols, default=num_cols[:5])

    method = st.sidebar.selectbox("이상 탐지 기법", ["Z-score", "PCA", "DBSCAN"])

    df_selected = df[selected_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected)

    st.subheader(f"이상 탐지 방법: {method}")

    if method == "Z-score":
        z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))
        threshold = st.sidebar.slider("Z-score threshold", 2.5, 5.0, 3.0, step=0.1)
        anomaly_mask = (z_scores > threshold).any(axis=1)
        df["anomaly"] = anomaly_mask.astype(int)
        st.write(f"🔍 이상치 탐지 결과: {anomaly_mask.sum()}개 발견")

    elif method == "PCA":
        n_comp = st.sidebar.slider("PCA 차원 수", 2, min(10, len(selected_features)), 2)
        pca = PCA(n_components=n_comp)
        pca_result = pca.fit_transform(X_scaled)
        df["pca_1"] = pca_result[:, 0]
        df["pca_2"] = pca_result[:, 1] if n_comp > 1 else 0

        fig, ax = plt.subplots()
        sns.scatterplot(x="pca_1", y="pca_2", data=df, alpha=0.6)
        st.pyplot(fig)

    elif method == "DBSCAN":
        eps = st.sidebar.slider("eps", 0.1, 5.0, 1.0, step=0.1)
        min_samples = st.sidebar.slider("min_samples", 3, 20, 5)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        df["anomaly"] = (labels == -1).astype(int)
        st.write(f"🔍 이상치 탐지 결과: {(labels == -1).sum()}개 발견")

        fig, ax = plt.subplots()
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette="Set1", alpha=0.6)
        st.pyplot(fig)

    # --- 이상 탐지 결과 표 ---
    st.subheader("이상 탐지 결과 미리보기")
    st.dataframe(df.head())

else:
    st.warning("좌측에서 CSV 파일을 업로드 해주세요.")
