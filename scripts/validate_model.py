import pandas as pd
import numpy as np
import json
import os
from model.thermal_model import ThermalModel
from utils.solar_utils import SolarUtility

# 1. Cargar Configuración
with open("config/plant_config.json") as f:
    config = json.load(f)

thermal = ThermalModel(config)
solar = SolarUtility(
    config['location']['latitude'],
    config['location']['longitude'],
    config['location']['altitude'],
    config['location']['timezone']
)

def run_validation(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el archivo {csv_path}. Ejecuta primero scripts/download_data.py si tienes acceso a la red.")
        return

    df = pd.read_csv(csv_path)
    df['_time'] = pd.to_datetime(df['_time'])
    
    results = []
    
    print(f"Validando modelo con {len(df)} puntos de datos...")
    
    for i, row in df.iterrows():
        # Obtener DNI estimada
        dni = solar.estimate_dni(row['RAD'], row['_time'])
        eta_opt = solar.get_fresnel_efficiency(
            row['_time'], 
            config['collector']['optical_efficiency_0'],
            config['collector']['iam_longitudinal'],
            config['collector']['iam_transversal']
        )
        
        # Estado actual (usando datos reales como entrada al paso t)
        # T_sf_in = T4 (después de bomba de recirculación)
        # T_sf_out = T8 (salida campo solar)
        # P_drum = P5
        current_state = {
            "P_drum": row['P5'],
            "T_sf_in": row['T4'],
            "T_sf_out": row['T8']
        }
        
        inputs = {
            "dni": dni,
            "t_amb": row['T_AMB'],
            "m_rec": row['CAUDAL'] / 3600.0, # Asumiendo m3/h -> kg/s aprox
            "m_feed": 0.1, # Valor dummy si no hay sensor exacto
            "m_vapor_real": row['CAUDAL VAP'] / 3600.0,
            "eta_opt": eta_opt
        }
        
        # Simular
        pred = thermal.simulate_step(current_state, inputs, dt_min=1)
        
        results.append({
            "time": row['_time'],
            "T8_real": row['T8'],
            "T8_sim": pred['T_sf_out_sim'],
            "Vap_real": row['CAUDAL VAP'],
            "Vap_sim": pred['m_vapor_gen'] * 3600.0,
            "Error_T": pred['T_sf_out_sim'] - row['T8']
        })
        
    res_df = pd.DataFrame(results)
    mae_t = res_df['Error_T'].abs().mean()
    print(f"MAE Temperatura (T8): {mae_t:.2f} °C")
    
    res_df.to_csv("data/validation_results.csv", index=False)
    print("Resultados de validación guardados en data/validation_results.csv")

if __name__ == "__main__":
    run_validation("data/plant_data_april.csv")
