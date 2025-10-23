import joblib
import numpy as np
import json
from pathlib import Path

class PipelineFadiga:
    def __init__(self, modelo_dir):
        modelo_dir = Path(modelo_dir)

        self.extrator = joblib.load(modelo_dir / "extrator.joblib")
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
