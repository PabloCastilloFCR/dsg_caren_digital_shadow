import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import json
import matplotlib.pyplot as plt
from model.thermal_model import ThermalModel
from model.pressure_controller import compute_valve_opening

# 1. Configuración de la prueba
with open("config/plant_config.json") as f:
    config = json.load(f)

model = ThermalModel(config)

# Parámetros sintéticos
DNI = 700.0
T_AMB = 20.0
P_SETPOINT = 2.513 # 1.5 bar manométrico + 1.013 bar amb
VALVE_OFFSET_BAR = 0.5 # Banda proporcional de la válvula (100% apertura a P_set + 0.5 bar)
NIVEL_DRUM = 200.0 # 200 mm
STEPS = 120 # 60 minutos
DT = 1 # 1 minuto por paso

# Estado inicial
# Inventario del drum SOLO (la cañería es un nodo aparte: T_pipe)
props_init = model.get_water_properties(1.0, T_C=T_AMB)
area_drum = np.pi * (model.drum_d/2)**2
vol_liq_drum_init = (NIVEL_DRUM / 1000) * area_drum

m_total = vol_liq_drum_init * props_init['rho']
u_total = (m_total * props_init['u']) + (model.drum_metal_mass * model.pipe_cp * T_AMB)

state = {
    "P_drum": 1.0,
    "T_drum": T_AMB,
    "T_pipe": T_AMB,
    "T_sf_in": T_AMB,
    "T_sf_out": T_AMB,
    "T_tube": T_AMB,
    "T_loop": T_AMB,
    "level": NIVEL_DRUM,
    "M_total": m_total,
    "U_total": u_total
}

# Estado de la válvula (apertura calculada por el controlador proporcional)
current_valve_pos = 0.0

results = []

print(f"Iniciando prueba sintética: DNI={DNI}, P_set={P_SETPOINT-1.013} bar g")

# Variable de estado para la bomba
feed_pump_on = False

for t in range(STEPS):
    # 1. Control de Nivel (Bomba de agua de alimentación) - Hysteresis
    if state['level'] < NIVEL_DRUM-5:
        feed_pump_on = True
    elif state['level'] > NIVEL_DRUM+30:
        feed_pump_on = False
    
    m_feed = 0.01 if feed_pump_on else 0.0 # ~0.6 LPM
    
    # 2. Control de presión: proporcional puro (lógica del fabricante)
    #    %_apertura = clamp((P_actual - P_setpoint) / offset, 0, 1) * 100
    current_valve_pos = compute_valve_opening(state['P_drum'], P_SETPOINT, VALVE_OFFSET_BAR)
    
    inputs = {
        "dni": DNI,
        "t_amb": T_AMB,
        "m_rec": 1.66,
        "m_feed": m_feed,
        "valve_open": current_valve_pos,
        "eta_opt": 0.25, # Ajustado para que Q_net sea coherente con 18kg/h
        "T_feed": 20.0,
        "debug": t % 15 == 0
    }
    
    # Simular paso con sub-steps para estabilidad
    SUB_STEPS = 10
    dt_sub = DT / SUB_STEPS
    
    for _ in range(SUB_STEPS):
        new_state_sim = model.simulate_step(state, inputs, dt_min=dt_sub)
        P_safe = max(1.1, min(10.0, new_state_sim['P_drum_sim']))
        
        state = {
            "P_drum": P_safe,
            "T_drum": new_state_sim['T_drum_sim'],
            "T_pipe": new_state_sim['T_pipe_sim'],
            "T_sf_in": new_state_sim['T_pipe_sim'], # El agua entra al SF desde el lazo de recirculación
            "T_sf_out": new_state_sim['T_sf_out_sim'],
            "T_tube": new_state_sim['T_tube_sim'],
            "T_loop": new_state_sim['T_loop_sim'],
            "level": new_state_sim['level_sim'],
            "M_total": new_state_sim['M_total_new'],
            "U_total": new_state_sim['U_total_new']
        }
    
    # Registro de datos
    results.append({
        "time": t,
        "P": state['P_drum'] - 1.013,
        "Level": state['level'],
        "T_drum": state['T_drum'],
        "T_pipe": state['T_pipe'],
        "T_in": state['T_sf_in'],
        "T_out": state['T_sf_out'],
        "Valve": current_valve_pos,
        "Vapor": new_state_sim['m_vapor_sim'] * 3600 # kg/h
    })

# Visualización
times = [r['time'] for r in results]
pressures = [r['P'] for r in results]
valves = [r['Valve'] for r in results]
vapors = [r['Vapor'] for r in results]
levels = [r['Level'] for r in results]
t_drum = [r['T_drum'] for r in results]
t_in = [r['T_in'] for r in results]
t_out = [r['T_out'] for r in results]

fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot 1: Presión y Válvula
ax[0].set_ylabel('Presión Manométrica (bar)', color='tab:blue')
ax[0].plot(times, pressures, label='Presión Drum', color='tab:blue', linewidth=2)
ax[0].axhline(y=P_SETPOINT-1.013, color='r', linestyle='--', label='Setpoint')
ax[0].tick_params(axis='y', labelcolor='tab:blue')
ax[0].legend(loc='upper left')

ax1b = ax[0].twinx()
ax1b.set_ylabel('Válvula (%)', color='tab:green')
ax1b.plot(times, valves, label='Apertura Válvula', color='tab:green', linestyle='--')
ax1b.tick_params(axis='y', labelcolor='tab:green')

# Plot 2: Temperaturas
ax[1].set_ylabel('Temperatura (°C)')
ax[1].plot(times, t_drum, label='T_drum (Saturación)', color='black', linewidth=2)
ax[1].plot(times, t_in, label='T_sf_in', color='tab:blue', linestyle=':')
ax[1].plot(times, t_out, label='T_sf_out', color='tab:red', linestyle='-.')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

# Plot 3: Nivel y Vapor
ax[2].set_ylabel('Nivel (mm)', color='tab:purple')
ax[2].plot(times, levels, label='Nivel Drum', color='tab:purple')
ax[2].tick_params(axis='y', labelcolor='tab:purple')
ax[2].set_xlabel('Tiempo (min)')

ax2b = ax[2].twinx()
ax2b.set_ylabel('Vapor (kg/h)', color='tab:orange')
ax2b.plot(times, vapors, label='Producción Vapor', color='tab:orange')
ax2b.tick_params(axis='y', labelcolor='tab:orange')

plt.suptitle('Prueba Sintética Digital Twin - Simulación Térmica Completa')
fig.tight_layout()
# plt.savefig("tests/synthetic_test_result.png") # Opcional: guardar copia
print("\nAbriendo ventana interactiva... Cierra la ventana para finalizar el script.")
plt.show()
