import streamlit as st

st.set_page_config(
    page_title="주조 공정 모니터링 대시보드",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ 주조 공정 센서 기반 모니터링 대시보드")
st.markdown("""
---
### 📋 대시보드 안내
왼쪽 사이드바에서 원하는 페이지를 선택하세요.

- 📊 공정 변수 이상 탐지  
- 🤖 불량 예측 시뮬레이터  
- 🔍 변수 영향도 및 예측 근거  
- 📈 실시간 공정 모니터링  
- 📋 종합 리포트 및 모델 성능  

---
""")
