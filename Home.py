import streamlit as st
from datetime import datetime
from utils.common import inject_css 
from utils.loader import load_sample_data


st.set_page_config(
    page_title="🏭 AI 품질관리 시스템",
    layout="wide",
    page_icon="🤖"
)



st.markdown("""
<div class="main-header">
    <h1>🤖 AI 기반 스마트 제조 품질관리 시스템</h1>
    <p>실시간 이상 탐지 • 품질 예측 • 설명 가능한 AI • 통합 모니터링</p>
</div>
""", unsafe_allow_html=True)

inject_css()
data = load_sample_data()

# 종합 대시보드
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 실시간 시스템 현황")

    # 주요 지표들
    quality_rate = data['quality'].tail(100).mean()
    avg_pressure = data['cast_pressure'].tail(24).mean()
    temp_variance = data[['upper_mold_temp1', 'lower_mold_temp1']].tail(24).var().mean()

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("품질 합격률", f"{quality_rate:.1%}", "↑ 2.3%")
    with metrics_col2:
        st.metric("평균 주조압력", f"{avg_pressure:.1f}", "↓ 5.2")
    with metrics_col3:
        st.metric("온도 안정성", f"{temp_variance:.1f}", "↑ 1.1")

with col2:
    st.subheader("🎯 주요 알림")
    st.markdown("""
    <div class="info-card">
        <strong>📈 품질 개선 권고</strong><br>
        주조압력을 120-130 범위로 조정하여 품질 향상 가능
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="normal-card">
        <strong>✅ 시스템 정상</strong><br>
        모든 센서가 정상 작동 중입니다
    </div>
    """, unsafe_allow_html=True)

# 빠른 품질 예측
st.subheader("⚡ 빠른 품질 체크")

quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
with quick_col1:
    q_pressure = st.number_input("주조압력", value=125.0)
with quick_col2:
    q_temp = st.number_input("상부온도", value=225.0)
with quick_col3:
    q_sleeve = st.number_input("슬리브온도", value=195.0)
with quick_col4:
    if st.button("🔍 빠른 예측"):
        # 간단한 예측 로직
        risk_score = 0
        if q_pressure > 200 or q_pressure < 100: risk_score += 30
        if q_temp > 300 or q_temp < 180: risk_score += 25
        if q_sleeve > 400: risk_score += 45
        
        if risk_score > 50:
            st.error("⚠️ 불량 위험 높음!")
        else:
            st.success("✅ 양품 예상")



st.sidebar.header("🔧 시스템 상태")
st.sidebar.success("✅ 모델 서버: 정상")
st.sidebar.success("✅ 데이터 수집: 정상")
st.sidebar.success("✅ 이상 탐지: 활성")
st.sidebar.info(f"🕐 마지막 업데이트: {datetime.now().strftime('%H:%M:%S')}")


# 푸터
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; margin-top: 2rem;'>
    <p>🤖 AI 품질관리 시스템 v2.0 | 
    📊 정확도: 98.86% | 
    🔄 실시간 업데이트 | 
    ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)