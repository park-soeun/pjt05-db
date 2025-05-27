import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats

class AnomalyDetector:
    """이상 탐지 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.is_fitted = False
    
    def fit(self, X):
        """이상 탐지 모델 훈련"""
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)
        
        self.isolation_forest.fit(X_scaled)
        self.dbscan.fit(X_pca)
        self.is_fitted = True
        
        return self
    
    def detect_anomalies(self, X):
        """이상 탐지 수행"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Z-score 기반 이상 탐지
        z_scores = np.abs(stats.zscore(X_scaled, axis=0))
        z_anomalies = (z_scores > 3).any(axis=1)
        
        # Isolation Forest 이상 탐지
        iso_anomalies = self.isolation_forest.predict(X_scaled) == -1
        
        # DBSCAN 이상 탐지
        dbscan_labels = self.dbscan.fit_predict(X_pca)
        dbscan_anomalies = dbscan_labels == -1
        
        # 종합 이상 점수 (0-1)
        anomaly_scores = (z_anomalies.astype(int) + 
                          iso_anomalies.astype(int) + 
                          dbscan_anomalies.astype(int)) / 3
        
        return {
            'anomaly_scores': anomaly_scores,
            'z_anomalies': z_anomalies,
            'iso_anomalies': iso_anomalies,
            'dbscan_anomalies': dbscan_anomalies,
            'is_anomaly': anomaly_scores > 0.5
        }
