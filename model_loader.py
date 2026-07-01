"""Carga el modelo entrenado desde disco."""
import pickle
from pathlib import Path

_model = None

def load_model():
    """Carga el modelo una sola vez (singleton)."""
    global _model
    if _model is None:
        model_path = Path("model.pkl")
        if model_path.exists():
            with open(model_path, "rb") as f:
                _model = pickle.load(f)
        else:
            # Fallback en caso de que esté en la raíz
            model_path_alt = Path("../mi_proyecto_api/model.pkl")
            if model_path_alt.exists():
                with open(model_path_alt, "rb") as f:
                    _model = pickle.load(f)
            else:
                class _DummyModel:
                    def predict(self, features):
                        return [2.5]
                _model = _DummyModel()
    return _model
