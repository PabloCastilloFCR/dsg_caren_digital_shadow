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
        },
        "piping": {
            "length_m": 60,
            "total_metal_mass_kg": 120,
            "fluid_volume_L": 80,
            "material_cp": 0.5
        }
    }
    
    model = ThermalModel(config)
    
    # Caso 1: Medio día, mucha radiación -> el lazo de recirculación (T_pipe) se calienta
    current_state = {
        "P_drum": 10.0,
        "T_drum": 100.0,
        "T_pipe": 100.0,
        "T_sf_in": 100.0,
        "T_sf_out": 100.0,
        "T_tube": 100.0,
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

    print(f"T_pipe_sim: {res['T_pipe_sim']:.2f} °C | Q_net: {res['Q_net']:.2f} kW")
    # Con sol y flujo, el receptor entrega calor al agua del lazo -> T_pipe sube
    assert res['Q_net'] > 0
    assert 100.0 < res['T_pipe_sim'] < 120.0
    print("Prueba Caso 1: PASADA")

    # Caso 2: Generación de vapor y cambio de presión
    current_state["T_sf_in"] = 175.0
    current_state["T_sf_out"] = 185.0
    current_state["P_drum"] = 10.0
    
    res = model.simulate_step(current_state, inputs, dt_min=1)
    print(f"Flujo vapor calculado: {res['m_vapor_sim']:.4f} kg/s")
    assert res['m_vapor_sim'] > 0
    print("Prueba Caso 2: PASADA")

    # Caso 3: Sin recirculacion, el calor del receptor no debe entrar al drum
    no_flow_state = {
        "P_drum": 1.013,
        "T_sf_in": 30.0,
        "T_sf_out": 30.0,
        "T_tube": 30.0,
        "level": 200.0,
        "M_total": 100.0,
        "U_total": 100.0 * 125.0
    }
    no_flow_inputs = {
        "dni": 1000.0,
        "t_amb": 30.0,
        "m_rec": 0.0,
        "m_feed": 0.0,
        "valve_open": 0.0,
        "eta_opt": 0.6
    }
    res = model.simulate_step(no_flow_state, no_flow_inputs, dt_min=1)
    assert res['Q_net'] == 0.0
    assert abs(res['U_total_new'] - no_flow_state['U_total']) < 1e-6
    print("Prueba Caso 3: PASADA")

    # Caso 4: La entalpia de entrada al campo usa T_sf_in, no liquido saturado
    cold_inlet_state = {
        "P_drum": 3.0,
        "T_sf_in": 60.0,
        "T_sf_out": 60.0,
        "T_tube": 60.0,
        "level": 200.0,
        "M_total": 100.0,
        "U_total": 100.0 * 250.0
    }
    cold_inlet_inputs = {
        "dni": 0.0,
        "t_amb": 60.0,
        "m_rec": 1.0,
        "m_feed": 0.0,
        "valve_open": 0.0,
        "eta_opt": 0.6
    }
    res = model.simulate_step(cold_inlet_state, cold_inlet_inputs, dt_min=1)
    assert abs(res['T_sf_out_sim'] - cold_inlet_state['T_sf_in']) < 0.1
    print("Prueba Caso 4: PASADA")

    # Caso 5: Un estado subenfriado no debe resolverse como saturado.
    props_30c = model.get_water_properties(1.013, T_C=30.0)
    # El solver de equilibrio usa solo el metal del drum como inercia residual.
    residual_metal_mass = model.drum_metal_mass
    subcooled_mass = 100.0
    subcooled_u = (subcooled_mass * props_30c['u']) + (residual_metal_mass * model.pipe_cp * 30.0)
    p_sub, t_sub, x_sub = model._solve_equilibrium(subcooled_mass, subcooled_u, P_hint_bar=1.013)
    assert abs(p_sub - 1.013) < 1e-6
    assert abs(t_sub - 30.0) < 0.1
    assert x_sub == 0.0
    print("Prueba Caso 5: PASADA")

    # Caso 6: El warmup reduce la energia absorbida efectiva cuando el tubo esta frio.
    config["collector"]["warmup"] = {
        "enabled": True,
        "start_temp_C": 40,
        "full_temp_C": 95,
        "min_factor": 0.25
    }
    warmup_model = ThermalModel(config)
    warmup_state = {
        "P_drum": 1.013,
        "T_sf_in": 30.0,
        "T_sf_out": 30.0,
        "T_tube": 30.0,
        "level": 200.0,
        "M_total": 100.0,
        "U_total": 100.0 * 125.0
    }
    warmup_inputs = {
        "dni": 1000.0,
        "t_amb": 30.0,
        "m_rec": 0.0,
        "m_feed": 0.0,
        "valve_open": 0.0,
        "eta_opt": 0.6
    }
    res = warmup_model.simulate_step(warmup_state, warmup_inputs, dt_min=1)
    assert abs(res['warmup_factor'] - 0.25) < 1e-9
    assert abs(res['Q_abs_effective'] - (res['Q_abs'] * 0.25)) < 1e-9
    print("Prueba Caso 6: PASADA")

    # Caso 7: La bateria termica puede descargar calor al fluido sin radiacion.
    config["collector"]["warmup"] = {"enabled": False}
    config["collector"]["thermal_battery"] = {
        "enabled": True,
        "capacity_kj_k": 1000,
        "charge_fraction": 0.2,
        "ua_fluid_w_k": 500,
        "ua_loss_w_k": 20
    }
    battery_model = ThermalModel(config)
    battery_state = {
        "P_drum": 1.013,
        "T_sf_in": 40.0,
        "T_sf_out": 40.0,
        "T_tube": 40.0,
        "T_loop": 90.0,
        "level": 200.0,
        "M_total": 100.0,
        "U_total": 100.0 * 170.0
    }
    battery_inputs = {
        "dni": 0.0,
        "t_amb": 20.0,
        "m_rec": 1.0,
        "m_feed": 0.0,
        "valve_open": 0.0,
        "eta_opt": 0.6
    }
    res = battery_model.simulate_step(battery_state, battery_inputs, dt_min=1)
    assert res['Q_abs'] == 0.0
    assert res['Q_battery_fluid'] > 0.0
    assert res['Q_net'] > 0.0
    assert res['T_loop_sim'] < battery_state['T_loop']
    print("Prueba Caso 7: PASADA")

if __name__ == "__main__":
    test_energy_balance()
