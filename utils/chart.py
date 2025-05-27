import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_anomaly_detection_charts(data, anomaly_results):
    """이상 탐지 시각화"""

    # 이상 점수 시계열
    fig1 = go.Figure()

    colors = ['red' if x else 'green' for x in anomaly_results['is_anomaly']]

    fig1.add_trace(go.Scatter(
        x=data['timestamp'],
        y=anomaly_results['anomaly_scores'],
        mode='markers+lines',
        marker=dict(color=colors, size=8),
        name='이상 점수',
        line=dict(width=2)
    ))

    fig1.add_hline(y=0.5, line_dash="dash", line_color="red",
                   annotation_text="이상 임계값")

    fig1.update_layout(
        title="🚨 실시간 이상 탐지 모니터링",
        xaxis_title="시간",
        yaxis_title="이상 점수",
        height=400,
        showlegend=True
    )

    # PCA 시각화
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    feature_cols = ['cast_pressure', 'upper_mold_temp1', 'lower_mold_temp1',
                    'sleeve_temperature', 'Coolant_temperature']
    X_scaled = scaler.fit_transform(data[feature_cols])
    X_pca = pca.fit_transform(X_scaled)

    fig2 = go.Figure()

    normal_mask = ~anomaly_results['is_anomaly']
    anomaly_mask = anomaly_results['is_anomaly']

    fig2.add_trace(go.Scatter(
        x=X_pca[normal_mask, 0],
        y=X_pca[normal_mask, 1],
        mode='markers',
        marker=dict(color='green', size=8, opacity=0.6),
        name='정상'
    ))

    fig2.add_trace(go.Scatter(
        x=X_pca[anomaly_mask, 0],
        y=X_pca[anomaly_mask, 1],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='이상'
    ))

    fig2.update_layout(
        title="🔍 PCA 기반 이상 탐지 시각화",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} 설명)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} 설명)",
        height=400
    )

    return fig1, fig2