"""Schemas Pydantic para predictor-band-gap."""
from pydantic import BaseModel

class InputData(BaseModel):
    electroneg_metal: float
    density: float
    coord_num: float

class PredictionResult(BaseModel):
    bandgap_predicted_eV: float
    status: str
    modelo_version: str
