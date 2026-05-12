import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from model.thermal_model import ThermalModel

def test_energy_balance():
    print("Iniciando prueba de balance de energía...")
    
    # Cargar config mínima
    config = {
        "collector": {
            "aperture_area": 100,
            "optical_efficiency_0": 0.6,
            "thermal_loss_coefficients": [0.5, 0.0]
        },
        "steam_drum": {
            "diameter_mm": 355.6,
            "height_mm": 672.0
        }
    }
    
    model = ThermalModel(config)
    
    # Caso 1: Medio día, mucha radiación
    # Inicialización de acumulación para la prueba
    # v = 0.0667 / 30 = 0.0022 m3/kg (aprox mezcla)
    current_state = {
        "P_drum": 10.0, 
        "T_sf_in": 100.0, 
        "T_sf_out": 100.0,
        "level": 200.0,
        "M_total": 30.0,
        "U_total": 30.0 * 420 # kJ (aprox liq saturado)
    }
    inputs = {
        "dni": 1000.0, 
        "t_amb": 20.0,
        "m_rec": 1.0,
        "m_feed": 0.05,
        "valve_open": 10.0,
        "eta_opt": 0.6
    }
    
    res = model.simulate_step(current_state, inputs, dt_min=1)
    
    print(f"T_out_sim calculada: {res['T_sf_out_sim']:.2f} °C")
    # T_out_sim debería ser similar a 113.39 (calculado antes)
    assert abs(res['T_sf_out_sim'] - 113.40) < 0.5
    print("Prueba Caso 1: PASADA")

    # Caso 2: Generación de vapor y cambio de presión
    current_state["T_sf_in"] = 175.0
    current_state["T_sf_out"] = 185.0
    current_state["P_drum"] = 10.0
    
    res = model.simulate_step(current_state, inputs, dt_min=1)
    print(f"Flujo vapor calculado: {res['m_vapor_sim']:.4f} kg/s")
    assert res['m_vapor_sim'] > 0
    print("Prueba Caso 2: PASADA")

if __name__ == "__main__":
    test_energy_balance()
