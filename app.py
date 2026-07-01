"""API FastAPI para predictor-band-gap."""
from fastapi import FastAPI, HTTPException
from schemas import InputData, PredictionResult
from model_loader import load_model
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pathlib import Path

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar modelo en startup
    load_model()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="predictor-band-gap",
    description="API de predicción de band gap de óxidos metálicos.",
    version="1.0.0",
)

@app.get("/health")
def health():
    """Verifica que el servicio está operativo."""
    return {"status": "ok", "servicio": "predictor-band-gap"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData):
    """Realiza una predicción con los datos de entrada."""
    model = load_model()
    try:
        # Extraer features
        diff = 3.44 - data.electroneg_metal
        features = [data.electroneg_metal, diff, data.density, data.coord_num]

        # Cargar scaler si existe
        scaler_path = Path("scaler_bandgap.pkl")
        if not scaler_path.exists():
            scaler_path = Path("../mi_proyecto_api/scaler_bandgap.pkl")

        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            features_scaled = scaler.transform([features])[0]
        else:
            features_scaled = features

        resultado = model.predict([features_scaled])[0]

        return PredictionResult(
            bandgap_predicted_eV=float(resultado),
            status="success",
            modelo_version="1.0.0"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
