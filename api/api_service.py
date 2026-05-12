import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from utils.solar_utils import SolarUtility
from model.thermal_model import ThermalModel

app = FastAPI(title="Digital Twin Fresnel DSG")

# Cargar Configuración
with open("config/plant_config.json") as f:
    config = json.load(f)

solar = SolarUtility(
    config['location']['latitude'],
    config['location']['longitude'],
    config['location']['altitude'],
    config['location']['timezone']
)
thermal = ThermalModel(config)

class PlantState(BaseModel):
    timestamp: str
    ghi: float
    t_amb: float
    p_drum: float
    level: float # Nivel en mm
    t_drum_bottom: float
    m_feed: float
    m_rec: float
    m_vapor: float
    valve_open: float # 0-100%
    t_sf_in: float
    t_sf_out: float

@app.get("/")
def read_root():
    return {"status": "Digital Twin Online", "plant": config['plant_name']}

@app.post("/predict")
def predict_state(state: PlantState):
    try:
        # 1. Estimar DNI y Eficiencia
        dni = solar.estimate_dni(state.ghi, state.timestamp)
        eta = solar.get_fresnel_efficiency(
            state.timestamp, 
            config['collector']['optical_efficiency_0'],
            config['collector']['iam_longitudinal'],
            config['collector']['iam_transversal']
        )
        
        # 2. Inicializar Masa y Energía Interna a partir del estado REAL
        # Esto permite que el gemelo se "resincronice" con la planta cada minuto
        props_f = thermal.get_water_properties(state.p_drum, x=0)
        props_g = thermal.get_water_properties(state.p_drum, x=1)
        
        # Calcular masa a partir de nivel
        area_drum = np.pi * (thermal.drum_d/2)**2
        vol_liq = (state.level / 1000.0) * area_drum
        vol_vap = thermal.drum_vol - vol_liq
        
        m_liq = vol_liq * props_f['rho']
        m_vap = vol_vap * props_g['rho']
        m_total = m_liq + m_vap
        u_total = m_liq * props_f['u'] + m_vap * props_g['u']
        
        # 3. Simular pasos de tiempo
        deltas = [1, 5, 10]
        predictions = {}
        
        for d in deltas:
            pred = thermal.simulate_step(
                current_state={
                    "P_drum": state.p_drum,
                    "T_sf_in": state.t_sf_in,
                    "T_sf_out": state.t_sf_out,
                    "level": state.level,
                    "M_total": m_total,
                    "U_total": u_total
                },
                inputs={
                    "dni": dni,
                    "t_amb": state.t_amb,
                    "m_feed": state.m_feed,
                    "m_rec": state.m_rec,
                    "valve_open": state.valve_open,
                    "eta_opt": eta
                },
                dt_min=d
            )
            predictions[f"{d}min"] = pred
            
        return {
            "current_calc": {
                "dni_estimated": dni,
                "optical_efficiency": eta
            },
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
