import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from model.thermal_model import ThermalModel

# 1. Configuración de la prueba
with open("config/plant_config.json") as f:
    config = json.load(f)

model = ThermalModel(config)

# 2. Cargar datos reales
try:
    df = pd.read_parquet('data/plant_data_processed.parquet')
    print("Datos cargados desde Parquet (zona horaria Chile).")
except Exception as e:
    print(f"Error al cargar Parquet: {e}. Cargando desde CSV...")
    df = pd.read_csv('data/plant_data_processed.csv')

# Asegurar formato datetime y convertir a local naive para ploteo
df['timestamp'] = pd.to_datetime(df['timestamp'])
if df['timestamp'].dt.tz is not None:
    # Convertimos a naive manteniendo la hora local para que matplotlib no la mueva a UTC
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

# 3. Parámetros de la simulación
P_SETPOINT = 2.0 + 1.013 # 2 bar manométrico + 1.013 bar amb
NIVEL_DRUM = 200.0 # 200 mm (valor por defecto)
DT = 1 # 1 minuto por paso (asumiendo que los datos están a 1 min)

# 4. Constantes Físicas y de Planta
M_REC_NOMINAL = 1.66 # Caudal recirculación nominal (kg/s)
H_VAP_WATER = 2257   # Calor latente de vaporización del agua (kJ/kg)
CP_WATER = 4.18      # Calor específico del agua (kJ/kgK)

# Estado inicial (usando el primer punto de datos)
first_row = df.iloc[0]
T_AMB_INIT = first_row['T_AMB']

props_init = model.get_water_properties(1.0, T_C=T_AMB_INIT)
area_drum = np.pi * (model.drum_d/2)**2
vol_liq_drum_init = (NIVEL_DRUM / 1000) * area_drum
vol_liq_total_init = vol_liq_drum_init + model.pipe_vol

# Masa total (Agua en drum + Agua en tubería)
m_total = vol_liq_total_init * props_init['rho']

# Energía total (Agua + Metal residual)
# Usamos pipe_metal_mass - tube_mass porque el tubo ahora tiene su propia inercia
residual_metal_mass_init = (model.pipe_metal_mass - model.tube_mass) + model.drum_metal_mass
u_total = (m_total * props_init['u']) + (residual_metal_mass_init * model.pipe_cp * T_AMB_INIT)

state = {
    "P_drum": 1.0,
    "T_drum": T_AMB_INIT,
    "T_sf_in": T_AMB_INIT,
    "T_sf_out": T_AMB_INIT,
    "T_tube": T_AMB_INIT,
    "level": NIVEL_DRUM,
    "M_total": m_total,
    "U_total": u_total
}

# Variables de estado y control
tracking_timer = 0
current_valve_pos = 0.0
integral_error = 0.0
kp = 50.0
ki = 0
valve_speed_limit = 100.0 
feed_pump_active = False

results = []

print(f"Iniciando simulación con datos reales. P_set={P_SETPOINT-1.013} bar g")

for i, row in df.iterrows():
    # Variables de entrada desde los datos
    GHI = row['RAD']
    T_AMB = row['T_AMB']
    
    # 1. Lógica de Seguimiento y Enfoque
    is_tracking = GHI > 200
    if is_tracking:
        tracking_timer += DT
    else:
        tracking_timer = 0
    
    # Los espejos solo concentran después de 5 minutos de seguimiento
    is_focused = tracking_timer >= 5
    dni_sim = GHI if is_focused else 0.0
    
    # 2. Control de Bombas (Alimentación y Recirculación)
    if is_tracking:
        # Histeresis de nivel: ON < 200mm, OFF > 250mm
        if state['level'] < 200.0:
            feed_pump_active = True
        elif state['level'] > 250.0:
            feed_pump_active = False
            
        m_feed = 0.01 if feed_pump_active else 0.0
        # Usar CAUDAL real si está disponible (convertir L/min a kg/s)
        m_rec_active = row['CAUDAL'] / 60.0 if row['CAUDAL'] > 0 else 0.0
    else:
        feed_pump_active = False
        m_feed = 0.0
        m_rec_active = 0.0
    
    # 3. Control de presión (PID) con umbral de activación (+0.2 bar)
    P_ACTIVATE = P_SETPOINT + 0.1
    if state['P_drum'] >= P_ACTIVATE or current_valve_pos > 0:
        error = state['P_drum'] - P_SETPOINT
        integral_error += error
        target_valve_open = max(0, min(100, kp * error + ki * integral_error))
        
        # Simular velocidad de movimiento de la válvula (Slew Rate)
        diff = target_valve_open - current_valve_pos
        step_move = max(-valve_speed_limit, min(valve_speed_limit, diff))
        current_valve_pos += step_move
        
        # Si la válvula se cierra completamente, resetear error integral
        if current_valve_pos <= 0:
            integral_error = 0.0
    else:
        target_valve_open = 0.0
        current_valve_pos = 0.0
        integral_error = 0.0
    
    inputs = {
        "timestamp": row['timestamp'],
        "dni": dni_sim,
        "t_amb": T_AMB,
        "m_rec": m_rec_active,
        "m_feed": m_feed,
        "valve_open": current_valve_pos,
        "eta_opt": config['collector']['optical_efficiency_0'] * 0.75,
        "T_feed": 20.0,
        "debug": i % 600 == 0
    }
    
    # Simular paso con sub-steps para estabilidad
    SUB_STEPS = 10
    dt_sub = DT / SUB_STEPS
    
    for _ in range(SUB_STEPS):
        new_state_sim = model.simulate_step(state, inputs, dt_min=dt_sub)
        P_safe = max(1.013, min(10.0, new_state_sim['P_drum_sim']))
        
        # Obtener T de saturación para el próximo paso
        props_f = model.get_water_properties(P_safe, x=0)
        
        state = {
            "P_drum": P_safe,
            "T_drum": new_state_sim['T_drum_sim'],
            "T_sf_in": props_f['T'] if P_safe > 1.02 else new_state_sim['T_drum_sim'],
            "T_sf_out": new_state_sim['T_sf_out_sim'],
            "T_tube": new_state_sim['T_tube_sim'],
            "level": new_state_sim['level_sim'],
            "M_total": new_state_sim['M_total_new'],
            "U_total": new_state_sim['U_total_new']
        }
    
    # Registro de datos
    t_drum_real = state['T_drum'] # Temperatura real (subenfriada o saturada)
    
    results.append({
        "timestamp": row['timestamp'],
        "RAD": GHI,
        "T_AMB": T_AMB,
        "P": state['P_drum'] - 1.013,
        "Level": state['level'],
        "T_drum": t_drum_real,
        "T_in": state['T_sf_in'],
        "T_out": state['T_sf_out'],
        "Valve": current_valve_pos,
        "Vapor": new_state_sim['m_vapor_sim'] * 3600, # kg/h
        # Datos Reales para comparación
        "P_real": row['P5'],
        "T_real": row['T5'],
        "T8_real": row['T8'],
        "T3_real": row['T3'],
        "T4_real": row['T4'],
        "Valve_real": row['EV FB'],
        "Vapor_real": row['CAUDAL VAP'],
        "Level_real": row['Nivel'],
        "CAUDAL": row['CAUDAL'],
        "DNI_est": new_state_sim.get('dni_est', 0.0),
        "eta_total": new_state_sim.get('eta_total', 0.0),
        "Q_net_sim": new_state_sim.get('Q_net', 0.0)
    })

# Visualización y Procesamiento de Resultados
res_df = pd.DataFrame(results)

# Cálculos de Ganancia Energética
# 1. Ganancia Real Estimada = m_rec_real * Cp * (T8 - T4)
res_df['m_rec_real'] = res_df['CAUDAL'] / 60.0
res_df['Gain_Real'] = res_df['m_rec_real'] * CP_WATER * (res_df['T8_real'] - res_df['T4_real'])
res_df['Gain_Real'] = res_df['Gain_Real'].clip(lower=0) 

# Filtrar anomalías en datos reales (P_thermal_peak < 50 kW)
res_df.loc[res_df['Gain_Real'] > 50, 'Gain_Real'] = 0

# 2. Ganancia Simulación = Q_net directo del modelo (kW)
res_df['Gain_Sim'] = res_df['Q_net_sim']

# FIGURA 1: Desempeño del Twin
fig1, ax = plt.subplots(5, 1, figsize=(12, 22), sharex=True)

# Plot 1: Radiación (GHI vs DNI est)
ax[0].plot(res_df['timestamp'], res_df['RAD'], label='GHI (Medido)', color='orange', alpha=0.6)
ax[0].plot(res_df['timestamp'], res_df['DNI_est'], label='DNI (Estimado Erbs)', color='red', linewidth=1.5)
ax[0].set_ylabel('Radiación (W/m2)')
ax[0].set_title('Radiación Solar')
ax[0].legend(loc='upper left')
ax0b = ax[0].twinx()
ax0b.plot(res_df['timestamp'], res_df['T_AMB'], label='T_amb (°C)', color='green', alpha=0.3)
ax0b.set_ylabel('Temp Amb (°C)')
ax0b.legend(loc='upper right')

# Plot 2: Presión y Válvula
ax[1].plot(res_df['timestamp'], res_df['P'], label='Presión Drum (Sim)', color='tab:blue', linewidth=2)
ax[1].axhline(y=P_SETPOINT-1.013, color='r', linestyle='--', label='Setpoint')
ax[1].set_ylabel('Presión Manométrica (bar)')
ax[1].set_title('Presión y Control')
ax[1].legend(loc='upper left')
ax1b = ax[1].twinx()
ax1b.plot(res_df['timestamp'], res_df['Valve'], label='Apertura Válvula (Sim)', color='tab:green', linestyle='--')
ax1b.set_ylabel('Válvula (%)')
ax1b.legend(loc='upper right')

# Plot 3: Temperaturas
ax[2].plot(res_df['timestamp'], res_df['T_drum'], label='T_drum (Sim Sat)', color='black', linewidth=2)
ax[2].plot(res_df['timestamp'], res_df['T_in'], label='T_sf_in (Sim)', color='tab:blue', linestyle=':')
ax[2].plot(res_df['timestamp'], res_df['T_out'], label='T_sf_out (Sim)', color='tab:red', linestyle='-.')
ax[2].set_ylabel('Temperatura (°C)')
ax[2].set_title('Temperaturas del Sistema')
ax[2].legend()
ax[2].grid(True, alpha=0.3)

# Plot 4: Nivel y Vapor
ax[3].plot(res_df['timestamp'], res_df['Level'], label='Nivel Drum (Sim)', color='tab:purple')
ax[3].set_ylabel('Nivel (mm)')
ax[3].set_title('Nivel y Producción')
ax[3].legend(loc='upper left')
ax3b = ax[3].twinx()
ax3b.plot(res_df['timestamp'], res_df['Vapor'], label='Producción Vapor (Sim)', color='tab:orange')
ax3b.set_ylabel('Vapor (kg/h)')
ax3b.legend(loc='upper right')

# Plot 5: Eficiencia Óptica (eta_opt * IAM)
ax[4].plot(res_df['timestamp'], res_df['eta_total'], label='Eficiencia Óptica Total (incl. IAM)', color='tab:cyan', linewidth=2)
ax[4].set_ylabel('Eficiencia (-)')
ax[4].set_title('Desempeño Óptico (ETA * IAM)')
ax[4].set_ylim(0, 1.0)
ax[4].legend()
ax[4].grid(True, alpha=0.3)

import matplotlib.dates as mdates

fig1.suptitle(f'Digital Twin: Desempeño Simulación - Setpoint: {P_SETPOINT-1.013} bar g')
ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig1.tight_layout()
fig1.savefig("tests/real_data_test_performance.png")

# FIGURA 2: Comparación Simulación vs Datos Reales
fig2, ax = plt.subplots(6, 1, figsize=(12, 26), sharex=True)

# Plot 1: Presión Drum (Sim vs Real)
ax[0].plot(res_df['timestamp'], res_df['P'], label='Presión Drum (Sim)', color='tab:blue', linewidth=2)
ax[0].plot(res_df['timestamp'], res_df['P_real'], label='Presión P5 (Real)', color='tab:cyan', linestyle='--')
ax[0].set_ylabel('Presión (bar g)')
ax[0].set_title('Comparación Presión')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Plot 2: Temperaturas (Sim vs Real T5, T8, T3, T4)
ax[1].plot(res_df['timestamp'], res_df['T_drum'], label='T_drum (Sim Sat)', color='black', linewidth=2)
ax[1].plot(res_df['timestamp'], res_df['T_real'], label='T5 (Real Drum)', color='tab:red', linestyle='--')
ax[1].plot(res_df['timestamp'], res_df['T8_real'], label='T8 (Real Salida Campo)', color='tab:orange', linestyle=':')
ax[1].plot(res_df['timestamp'], res_df['T3_real'], label='T3 (Real Retorno)', color='tab:brown', linestyle='-.', alpha=0.6)
ax[1].plot(res_df['timestamp'], res_df['T4_real'], label='T4 (Real Entrada Campo)', color='tab:blue', linestyle='--', alpha=0.6)
ax[1].set_ylabel('Temperatura (°C)')
ax[1].set_title('Comparación Temperaturas y Pérdidas en Retorno')
ax[1].legend(loc='lower right', fontsize='small', ncol=2)
ax[1].grid(True, alpha=0.3)

# Plot 3: Ganancia Energética (Sim vs Real Estimada)
ax[2].plot(res_df['timestamp'], res_df['Gain_Sim'], label='Ganancia Neta Sim (kW)', color='tab:red', linewidth=2)
ax[2].plot(res_df['timestamp'], res_df['Gain_Real'], label='Ganancia Real Estimada (kW)', color='tab:green', linestyle='--')
ax[2].set_ylabel('Potencia (kW)')
ax[2].set_title('Balance de Energía: Ganancia Térmica del Campo')
ax[2].legend()
ax[2].grid(True, alpha=0.3)

# Plot 4: Nivel Drum (Sim vs Real)
ax[3].plot(res_df['timestamp'], res_df['Level'], label='Nivel (Sim)', color='tab:purple', linewidth=2)
ax[3].plot(res_df['timestamp'], res_df['Level_real'], label='Nivel (Real)', color='tab:pink', linestyle='--')
ax[3].set_ylabel('Nivel (mm)')
ax[3].set_title('Comparación Nivel Drum')
ax[3].legend()
ax[3].grid(True, alpha=0.3)

# Plot 5: Apertura Válvula (Sim vs Real)
ax[4].plot(res_df['timestamp'], res_df['Valve'], label='Válvula (Sim PID)', color='tab:green', linewidth=2)
ax[4].plot(res_df['timestamp'], res_df['Valve_real'], label='EV FB (Real)', color='tab:olive', linestyle='--')
ax[4].set_ylabel('Apertura (%)')
ax[4].set_title('Comparación Control Válvula')
ax[4].legend()
ax[4].grid(True, alpha=0.3)

# Plot 6: Flujo Vapor (Sim vs Real)
ax[5].plot(res_df['timestamp'], res_df['Vapor'], label='Vapor (Sim)', color='tab:orange', linewidth=2)
ax[5].plot(res_df['timestamp'], res_df['Vapor_real'], label='CAUDAL VAP (Real)', color='brown', linestyle='--')
ax[5].set_ylabel('Flujo Vapor (kg/h)')
ax[5].set_title('Comparación Producción Vapor')
ax[5].legend()
ax[5].grid(True, alpha=0.3)

fig2.suptitle('Validación: Simulación vs Datos Reales de Planta')
ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig2.tight_layout()
fig2.savefig("tests/real_data_test_validation.png")

print("\nGráficos guardados en:")
print("- tests/real_data_test_performance.png")
print("- tests/real_data_test_validation.png")
plt.show()
