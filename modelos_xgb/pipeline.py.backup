import joblib
import numpy as np
import json
from pathlib import Path
from scipy import stats

class ExtratorFeatures:
    def __init__(self):
        nomes_sinais = ['PERCLOS', 'MAR', 'BLINK_RATE', 'HEAD_STABILITY']
        nomes_stats = ['mean', 'std', 'median', 'min', 'max', 'range', 'q25', 'q75', 'trend', 'zcr', 'autocorr']
        
        self.feature_names = []
        for sinal in nomes_sinais:
            for stat in nomes_stats:
                self.feature_names.append(f"{sinal}_{stat}")
    
    def trend_slope(self, sinal):
        if len(sinal) < 2:
            return 0.0
        x = np.arange(len(sinal))
        try:
            slope, _, _, _, _ = stats.linregress(x, sinal)
            return slope if not np.isnan(slope) else 0.0
        except:
            return 0.0
    
    def zero_crossing_rate(self, sinal):
        if len(sinal) < 2:
            return 0.0
        mean_centered = sinal - np.mean(sinal)
        crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        return crossings / len(sinal)
    
    def autocorr_lag1(self, sinal):
        if len(sinal) < 3:
            return 0.0
        try:
            corr = np.corrcoef(sinal[:-1], sinal[1:])[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def extrair_features_sinal(self, sinal):
        features = [
            np.mean(sinal), np.std(sinal), np.median(sinal),
            np.min(sinal), np.max(sinal), np.ptp(sinal),
            np.percentile(sinal, 25), np.percentile(sinal, 75),
            self.trend_slope(sinal), self.zero_crossing_rate(sinal), self.autocorr_lag1(sinal)
        ]
        return features
    
    def transform(self, X_sequences):
        n_samples = X_sequences.shape[0]
        n_features = len(self.feature_names)
        X_features = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            sequence = X_sequences[i]
            sample_features = []
            
            for signal_idx in range(4):
                sinal = sequence[:, signal_idx]
                features_sinal = self.extrair_features_sinal(sinal)
                sample_features.extend(features_sinal)
            
            X_features[i] = sample_features
        
        return X_features

class PipelineFadiga:
    def __init__(self, modelo_dir):
        modelo_dir = Path(modelo_dir)
        
        # Cria extrator ao invés de carregar do arquivo
        self.extrator = ExtratorFeatures()
        self.scaler = joblib.load(modelo_dir / "scaler.joblib")
        self.modelo = joblib.load(modelo_dir / "modelo_xgb.joblib")
        
        with open(modelo_dir / "info_classes.json", "r") as f:
            self.class_info = json.load(f)
    
    def predict_sequence(self, sequence):
        """Prediz fadiga de uma sequence (90, 4)"""
        sequences = np.expand_dims(sequence, axis=0)
        features = self.extrator.transform(sequences)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.modelo.predict(features_scaled)[0]
        probabilities = self.modelo.predict_proba(features_scaled)[0]
        
        prob_dict = {
            "Alerta": float(probabilities[0]),
            "Sonolento": float(probabilities[1])
        }
        
        class_name = "Alerta" if prediction == 0 else "Sonolento"
        
        return prediction, prob_dict, class_name