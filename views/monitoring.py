import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import load_sample_data

def create_realtime_monitoring():
    """실시간 모니터링 대시보드"""
    
    st.subheader("📡 실시간 공정 모니터링")
    
    data = load_sample_data()
    latest_data = data.tail(24)
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_quality_rate = latest_data['quality'].mean()
    current_anomaly_count = len(latest_data) - int(current_quality_rate * len(latest_data))
    avg_cycle_time = latest_data['production_cycletime'].mean()
    efficiency = min(100, (40 / avg_cycle_time) * 100)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>✅ 품질 합격률</h4>
            <h2>{current_quality_rate:.1%}</h2>
            <p>최근 24시간</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="{'anomaly-card' if current_anomaly_count > 2 else 'metric-card'}">
            <h4>🚨 이상 감지 건수</h4>
            <h2>{current_anomaly_count}건</h2>
            <p>최근 24시간</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ 평균 사이클타임</h4>
            <h2>{avg_cycle_time:.1f}초</h2>
            <p>목표: 40초</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="{'normal-card' if efficiency > 90 else 'info-card'}">
            <h4>⚡ 생산 효율성</h4>
            <h2>{efficiency:.1f}%</h2>
            <p>실시간 계산</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_quality = go.Figure()
        hourly_quality = latest_data.groupby(latest_data['timestamp'].dt.hour)['quality'].mean()
        
        fig_quality.add_trace(go.Scatter(
            x=hourly_quality.index,
            y=hourly_quality.values,
            mode='lines+markers',
            line=dict(width=3, color='green'),
            marker=dict(size=8),
            name='품질 합격률'
        ))
        
        fig_quality.add_hline(y=0.95, line_dash="dash", line_color="red", 
                              annotation_text="목표 95%")
        
        fig_quality.update_layout(
            title="📈 시간별 품질 합격률 트렌드",
            xaxis_title="시간",
            yaxis_title="합격률",
            height=400
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        fig_variables = go.Figure()
        variables = ['cast_pressure', 'upper_mold_temp1', 'sleeve_temperature']
        colors = ['blue', 'red', 'orange']
        
        for var, color in zip(variables, colors):
            normalized_vals = (latest_data[var] - latest_data[var].min()) / (latest_data[var].max() - latest_data[var].min())
            
            fig_variables.add_trace(go.Scatter(
                x=latest_data['timestamp'],
                y=normalized_vals,
                mode='lines',
                name=var,
                line=dict(color=color, width=2)
            ))
        
        fig_variables.update_layout(
            title="📊 주요 공정 변수 트렌드 (정규화)",
            xaxis_title="시간",
            yaxis_title="정규화된 값",
            height=400
        )
        st.plotly_chart(fig_variables, use_container_width=True)
    
    if st.checkbox("📋 상세 데이터 보기"):
        st.subheader("최근 공정 데이터")
        
        display_data = latest_data.tail(10).copy()
        display_data['품질'] = display_data['quality'].map({1: '✅ 양품', 0: '❌ 불량'})
        display_data['시간'] = display_data['timestamp'].dt.strftime('%H:%M')
        
        display_cols = ['시간', '품질', 'cast_pressure', 'upper_mold_temp1', 
                        'sleeve_temperature', 'production_cycletime']
        
        st.dataframe(
            display_data[display_cols].round(2),
            use_container_width=True,
            height=400
        )
