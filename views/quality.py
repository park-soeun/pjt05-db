import streamlit as st
import numpy as np
import plotly.graph_objects as go

def create_quality_prediction_interface(models):
    """품질 예측 인터페이스"""
    
    st.subheader("🎯 실시간 품질 예측")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**주요 공정 변수**")
        cast_pressure = st.slider("주조압력", 80.0, 300.0, 125.0, 1.0)
        upper_temp = st.slider("상부몰드온도", 150.0, 350.0, 225.0, 1.0)
        lower_temp = st.slider("하부몰드온도", 150.0, 350.0, 220.0, 1.0)
        sleeve_temp = st.slider("슬리브온도", 150.0, 500.0, 195.0, 1.0)
        coolant_temp = st.slider("냉각수온도", 15.0, 40.0, 25.0, 0.5)
    
    with col2:
        st.write("**보조 공정 변수**")
        low_speed = st.slider("저속구간속도", 0.5, 5.0, 2.5, 0.1)
        cycle_time = st.slider("생산사이클타임", 25, 60, 40, 1)
        molten_vol = st.slider("용탕량", 15.0, 40.0, 25.0, 0.5)
        strength = st.slider("물리강도", 250.0, 500.0, 375.0, 1.0)
    
    # 예측 버튼
    if st.button("🔍 품질 예측 실행", type="primary"):
        
        input_data = np.array([[cast_pressure, upper_temp, lower_temp, sleeve_temp, 
                                coolant_temp, low_speed, cycle_time, molten_vol, strength]])
        
        prediction = models['classification_model'].predict(input_data)[0]
        probability = models['classification_model'].predict_proba(input_data)[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="normal-card">
                    <h3>✅ 예측 결과: 양품</h3>
                    <p><strong>합격 확률:</strong> {probability[1]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="anomaly-card">
                    <h3>❌ 예측 결과: 불량품</h3>
                    <p><strong>불합격 확률:</strong> {probability[0]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "합격 확률 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            try:
                importances = models['classification_model'].feature_importances_
                feature_names = ['주조압력', '상부온도', '하부온도', '슬리브온도', 
                                 '냉각수온도', '저속', '사이클', '용탕량', '강도']
                
                fig_imp = go.Figure(go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation='h',
                    marker_color='skyblue'
                ))
                fig_imp.update_layout(
                    title="특성 중요도",
                    height=300,
                    margin=dict(l=100)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            except:
                st.info("특성 중요도를 계산할 수 없습니다.")
