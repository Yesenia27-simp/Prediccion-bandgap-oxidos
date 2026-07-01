"""API FastAPI y Frontend Web para predictor-band-gap."""
from fastapi import FastAPI
from schemas import InputData, PredictionResult
from model_loader import load_model
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pathlib import Path
import gradio as gr

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="predictor-band-gap",
    description="API de predicción de band gap de óxidos metálicos.",
    version="1.0.0",
)

# ==========================================
# 1. LÓGICA DE PREDICCIÓN (Tu código original)
# ==========================================
def realizar_prediccion(electroneg_metal, density, coord_num):
    model = load_model()
    diff = 3.44 - electroneg_metal
    features = [electroneg_metal, diff, density, coord_num]
    
    scaler_path = Path("scaler_bandgap.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform([features])[0]
    else:
        features_scaled = features
        
    resultado = model.predict([features_scaled])[0]
    return f"{resultado:.4f} eV"

# ==========================================
# 2. INTERFAZ GRÁFICA (Lo nuevo y bonito)
# ==========================================
interfaz_web = gr.Interface(
    fn=realizar_prediccion,
    inputs=[
        gr.Slider(minimum=0.5, maximum=4.0, step=0.1, label="Electronegatividad del Metal"),
        gr.Number(label="Densidad del Óxido (g/cm³)"),
        gr.Number(label="Número de Coordinación")
    ],
    outputs=gr.Textbox(label="Band Gap Predicho para Celda Solar"),
    title="☀️ Predicción de Band Gap en Óxidos Metálicos",
    description="Introduce las propiedades fisicoquímicas del material para calcular su viabilidad en aplicaciones de energía solar utilizando Machine Learning.",
    theme=gr.themes.Soft() # Este es el secreto para que se vea moderno y suave
)

# ==========================================
# 3. ENDPOINTS ORIGINALES Y MONTAJE
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok", "servicio": "predictor-band-gap"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData):
    res_texto = realizar_prediccion(data.electroneg_metal, data.density, data.coord_num)
    valor_float = float(res_texto.replace(" eV", ""))
    return PredictionResult(
        bandgap_predicted_eV=valor_float,
        status="success",
        modelo_version="1.0.0"
    )

# Aquí fusionamos la interfaz visual con tu servidor FastAPI
app = gr.mount_gradio_app(app, interfaz_web, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
