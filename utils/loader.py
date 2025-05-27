import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                          end=datetime.now(), freq='H')
    n_samples = len(dates)

    data = {
        'timestamp': dates,
        'cast_pressure': np.random.normal(125, 15, n_samples),
        'upper_mold_temp1': np.random.normal(225, 20, n_samples),
        'lower_mold_temp1': np.random.normal(220, 18, n_samples),
        'sleeve_temperature': np.random.normal(195, 25, n_samples),
        'Coolant_temperature': np.random.normal(25, 3, n_samples),
        'low_section_speed': np.random.normal(2.5, 0.5, n_samples),
        'production_cycletime': np.random.normal(40, 5, n_samples),
        'molten_volume': np.random.normal(25, 3, n_samples),
        'physical_strength': np.random.normal(375, 30, n_samples)
    }

    anomaly_indices = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
    data['cast_pressure'][anomaly_indices] += np.random.normal(100, 20, len(anomaly_indices))
    data['sleeve_temperature'][anomaly_indices] += np.random.normal(200, 50, len(anomaly_indices))

    quality_scores = []
    for i in range(n_samples):
        score = 0
        if 100 <= data['cast_pressure'][i] <= 150: score += 2
        if abs(data['upper_mold_temp1'][i] - data['lower_mold_temp1'][i]) < 15: score += 1
        if data['sleeve_temperature'][i] < 400: score += 1
        if data['physical_strength'][i] > 350: score += 1
        score += np.random.choice([-1, 0, 1], p=[0.1, 0.7, 0.2])
        quality_scores.append(1 if score >= 3 else 0)

    data['quality'] = quality_scores
    return pd.DataFrame(data)

@st.cache_resource
def load_models():
    """모델들 로드 또는 생성"""
    try:
        timestamp = "20250527_153117"
        model_path = f"models/random_forest_model_{timestamp}.joblib"
        model = joblib.load(model_path)

        preprocessing_path = f"models/preprocessing_{timestamp}.pkl"
        with open(preprocessing_path, 'rb') as f:
            preprocessing_data = pickle.load(f)

        return {
            'classification_model': model,
            'feature_columns': preprocessing_data['feature_columns'],
            'label_encoders': preprocessing_data['label_encoders']
        }
    except:
        st.warning("기존 모델을 찾을 수 없습니다. 샘플 데이터로 새 모델을 생성합니다.")
        return create_new_models()

def create_new_models():
    """새로운 모델들 생성"""
    data = load_sample_data()
    feature_cols = ['cast_pressure', 'upper_mold_temp1', 'lower_mold_temp1',
                    'sleeve_temperature', 'Coolant_temperature', 'low_section_speed',
                    'production_cycletime', 'molten_volume', 'physical_strength']
    X = data[feature_cols]
    y = data['quality']

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    return {
        'classification_model': rf_model,
        'feature_columns': feature_cols,
        'label_encoders': {}
    }
