import streamlit as st
from utils.loader import load_sample_data
from utils.detector import AnomalyDetector
from utils.chart import create_anomaly_detection_charts
from utils.common import inject_css 

inject_css()
data = load_sample_data()
detector = AnomalyDetector()


# 이상 탐지 모델 생성 및 적용

feature_cols = ['cast_pressure', 'upper_mold_temp1', 'lower_mold_temp1', 
                'sleeve_temperature', 'Coolant_temperature']
X = data[feature_cols]

detector.fit(X)
anomaly_results = detector.detect_anomalies(X)

# 이상 탐지 요약
total_anomalies = sum(anomaly_results['is_anomaly'])
anomaly_rate = total_anomalies / len(data) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("총 이상 건수", total_anomalies)
with col2:
    st.metric("이상 탐지율", f"{anomaly_rate:.1f}%")
with col3:
    latest_status = "이상" if anomaly_results['is_anomaly'][-1] else "정상"
    st.metric("현재 상태", latest_status)

# 이상 탐지 차트들
fig1, fig2 = create_anomaly_detection_charts(data, anomaly_results)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

# 이상 탐지 상세 정보
st.subheader("🔍 이상 탐지 상세 분석")

method_col1, method_col2, method_col3 = st.columns(3)

with method_col1:
    z_count = sum(anomaly_results['z_anomalies'])
    st.markdown(f"""
    <div class="metric-card">
        <h4>📊 Z-Score 기반</h4>
        <h3>{z_count}건 탐지</h3>
        <p>통계적 이상치 감지</p>
    </div>
    """, unsafe_allow_html=True)

with method_col2:
    iso_count = sum(anomaly_results['iso_anomalies'])
    st.markdown(f"""
    <div class="metric-card">
        <h4>🌲 Isolation Forest</h4>
        <h3>{iso_count}건 탐지</h3>
        <p>머신러닝 기반 감지</p>
    </div>
    """, unsafe_allow_html=True)

with method_col3:
    dbscan_count = sum(anomaly_results['dbscan_anomalies'])
    st.markdown(f"""
    <div class="metric-card">
        <h4>🎯 DBSCAN 클러스터링</h4>
        <h3>{dbscan_count}건 탐지</h3>
        <p>밀도 기반 이상 감지</p>
    </div>
    """, unsafe_allow_html=True)