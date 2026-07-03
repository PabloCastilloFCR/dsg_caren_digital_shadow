import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model.plant_simulator import PlantSimulator

# ── Configuración ──────────────────────────────────────────────────────────────
# Cambia esta fecha para comparar distintos días (formato YYYY-MM-DD)
DATES = [
    "2026-04-22",
    "2026-05-14",
    "2026-06-14",
    "2026-04-13",
]
SIM_DATE = DATES[0]

with open("config/plant_config_v2.json") as f:
    config = json.load(f)

# ── Cargar datos reales ────────────────────────────────────────────────────────
try:
    df = pd.read_parquet(f"data/processed_{SIM_DATE}.parquet")
    print(f"Datos cargados: processed_{SIM_DATE}.parquet  ({len(df)} filas)")
except Exception:
    df = pd.read_csv(f"data/processed_{SIM_DATE}.csv")
    print(f"Datos cargados: processed_{SIM_DATE}.csv  ({len(df)} filas)")

df['timestamp'] = pd.to_datetime(df['timestamp'])
if df['timestamp'].dt.tz is not None:
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

# ── Simular ────────────────────────────────────────────────────────────────────
P_SET_G = 2  # bar g — setpoint de presión del drum
sim    = PlantSimulator(config, P_SET_G)
res_df = sim.run(df)
print(f"Simulación completa: {len(res_df)} pasos")

# ── Desfase de parada de la bomba ─────────────────────────────────────────────
FLOW_THRESHOLD = 10.0   # L/min — por debajo de esto se considera "detenido"
sim_on  = res_df[res_df['CAUDAL_sim'] > FLOW_THRESHOLD]
real_on = res_df[res_df['CAUDAL']    > FLOW_THRESHOLD]
if not sim_on.empty and not real_on.empty:
    t_stop_sim  = sim_on['timestamp'].iloc[-1]
    t_stop_real = real_on['timestamp'].iloc[-1]
    delta_min   = (t_stop_real - t_stop_sim).total_seconds() / 60.0
    sign        = "despues" if delta_min >= 0 else "antes"

    # Condiciones en el momento de parada real
    stop_row = res_df[res_df['timestamp'] == t_stop_real].iloc[0]

    print(f"\nParada de bomba:")
    print(f"  Simulacion : {t_stop_sim.strftime('%H:%M')}")
    print(f"  Real       : {t_stop_real.strftime('%H:%M')}")
    print(f"  dt         : {abs(delta_min):.0f} min  (real para {sign} que sim)")
    print(f"\n  Condiciones en parada real ({t_stop_real.strftime('%H:%M')}):")
    print(f"    GHI      = {stop_row['RAD']:.1f} W/m2")
    print(f"    T5 (drum)= {stop_row['T5_real']:.1f} C")
    print(f"    T8 (campo)= {stop_row['T8_real']:.1f} C")
    print(f"    CAUDAL   = {stop_row['CAUDAL']:.1f} L/min")

# ── Guardar análisis de ganancia ───────────────────────────────────────────────
gain_cols = [
    'timestamp', 'RAD', 'DNI_est', 'T_AMB',
    # Presión
    'P5_real', 'P_sim',
    # Temperaturas zona solar
    'T5_real', 'T_drum_sim', 'T8_real', 'T8_sim',
    # Temperaturas zona fría
    'T3_real', 'T4_real', 'T4_sim',
    # Nivel
    'Level_real', 'Level_sim',
    # Caudal y válvula
    'CAUDAL', 'CAUDAL_sim', 'Valve_real', 'Valve_sim',
    # Vapor
    'Vapor_real', 'Vapor_sim',
    # Ganancia térmica
    'P4_real', 'P8_real',
    'Q_abs', 'Q_return', 'Q_gain_sun', 'Q_gain_ent', 'Q_loss_tube',
    'Gain_Real', 'Gain_Real_Cp', 'Gain_Real_h', 'Gain_Sim_Sun', 'Gain_Sim_Ent',
    'X_field_sim',
]
res_df[gain_cols].to_csv(f"tests/energy_gain_analysis_{SIM_DATE}.csv", index=False)

# ── Visualización: grilla 5×2 ─────────────────────────────────────────────────
# Fila 0: Radiación          | Presión
# Fila 1: T zona solar (T8)  | T zona fría (T4)
# Fila 2: Nivel              | Caudal
# Fila 3: Válvula            | Vapor
# Fila 4: Ganancia térmica   | Potencia absorbida (desglose)
fig, axes = plt.subplots(5, 2, figsize=(18, 27))
fig.subplots_adjust(hspace=0.45, wspace=0.3)
ax0, ax1 = axes[0]
ax2, ax3 = axes[1]
ax4, ax5 = axes[2]
ax6, ax7 = axes[3]
ax8, ax9 = axes[4]

# Convención visual: real = línea sólida, simulado = línea punteada
# ax0: Radiación y T_amb
ax0.plot(res_df['timestamp'], res_df['RAD'],     label='GHI (Med.)',   color='orange', linewidth=1.5)
ax0.plot(res_df['timestamp'], res_df['DNI_est'], label='DNI (DIRINT)', color='red',    linestyle='--', linewidth=1.5)
ax0.set_ylabel('Irradiancia (W/m²)'); ax0.set_title('Radiación Solar')
ax0.legend(fontsize='small'); ax0.grid(True, alpha=0.3)
ax0b = ax0.twinx()
ax0b.plot(res_df['timestamp'], res_df['T_AMB'], color='green', alpha=0.6, linewidth=1, label='T_amb')
ax0b.set_ylabel('T_amb (°C)', color='green'); ax0b.tick_params(axis='y', labelcolor='green')
ax0b.legend(fontsize='small', loc='upper right')

# ax1: Presión drum
ax1.plot(res_df['timestamp'], res_df['P5_real'], label='P5 (Real)',     color='tab:blue', linewidth=1.5)
ax1.plot(res_df['timestamp'], res_df['P_sim'],   label='Presión (Sim)', color='tab:blue', linestyle='--', linewidth=1.5)
ax1.axhline(y=P_SET_G, color='r', linestyle=':', label=f'Setpoint ({P_SET_G:.1f} bar g)', alpha=0.7)
ax1.set_ylabel('Presión (bar g)'); ax1.set_title('Presión Steam Drum')
ax1.legend(fontsize='small'); ax1.grid(True, alpha=0.3)

# ax2: Zona solar — T5 (drum) en azul, T8 (campo) en rojo
# Real: sólido | Simulado: punteado
ax2.plot(res_df['timestamp'], res_df['T5_real'],    label='T5 (Real Drum)',  color='tab:blue', linewidth=1.5)
ax2.plot(res_df['timestamp'], res_df['T_drum_sim'], label='T_drum (Sim)',    color='tab:blue', linestyle='--', linewidth=1.5)
ax2.plot(res_df['timestamp'], res_df['T8_real'],    label='T8 (Real Campo)', color='tab:red',  linewidth=1.5)
ax2.plot(res_df['timestamp'], res_df['T8_sim'],     label='T8 (Sim campo)',  color='tab:red',  linestyle='--', linewidth=1.5)
ax2.set_ylabel('Temperatura (°C)'); ax2.set_title('Zona Solar: T5/Drum (azul) y T8/Campo (rojo)')
ax2.legend(fontsize='small'); ax2.grid(True, alpha=0.3)

# ax3: Zona fría — entrada campo / post-bomba (≈ T4)
ax3.plot(res_df['timestamp'], res_df['T4_real'],   label='T4 (Real Entrada)',   color='tab:cyan',  linewidth=1.5)
ax3.plot(res_df['timestamp'], res_df['T3_real'],   label='T3 (Real Retorno)',   color='tab:brown', linewidth=1.5)
ax3.plot(res_df['timestamp'], res_df['T4_sim'],    label='T4 (Sim post-bomba)', color='tab:cyan',  linestyle='--', linewidth=1.5)
ax3.set_ylabel('Temperatura (°C)'); ax3.set_title('Zona Fría: T4 y T3 (entrada campo / bombas)')
ax3.legend(fontsize='small'); ax3.grid(True, alpha=0.3)

# ax4: Nivel drum
ax4.plot(res_df['timestamp'], res_df['Level_real'], label='Nivel (Real)', color='tab:purple', linewidth=1.5)
ax4.plot(res_df['timestamp'], res_df['Level_sim'],  label='Nivel (Sim)',  color='tab:purple', linestyle='--', linewidth=1.5)
ax4.set_ylabel('Nivel (mm)'); ax4.set_title('Nivel Steam Drum')
ax4.legend(fontsize='small'); ax4.grid(True, alpha=0.3)

# ax5: Caudal real vs simulado
ax5.plot(res_df['timestamp'], res_df['CAUDAL'],     label='Caudal (Real)', color='tab:green', linewidth=1.5)
ax5.plot(res_df['timestamp'], res_df['CAUDAL_sim'], label='Caudal (Sim)',  color='tab:green', linestyle='--', linewidth=1.5)
ax5.set_ylabel('Caudal (L/min)'); ax5.set_title('Caudal Recirculación')
ax5.legend(fontsize='small'); ax5.grid(True, alpha=0.3)

# ax6: Válvula
ax6.plot(res_df['timestamp'], res_df['Valve_real'], label='EV FB (Real)',  color='tab:olive', linewidth=1.5)
ax6.plot(res_df['timestamp'], res_df['Valve_sim'],  label='Válvula (Sim)', color='tab:olive', linestyle='--', linewidth=1.5)
ax6.set_ylabel('Apertura (%)'); ax6.set_title('Control Válvula Vapor')
ax6.legend(fontsize='small'); ax6.grid(True, alpha=0.3)

# ax7: Vapor
ax7.plot(res_df['timestamp'], res_df['Vapor_real'], label='CAUDAL VAP (Real)', color='tab:orange', linewidth=1.5)
ax7.plot(res_df['timestamp'], res_df['Vapor_sim'],  label='Vapor (Sim)',       color='tab:orange', linestyle='--', linewidth=1.5)
ax7.set_ylabel('Flujo Vapor (kg/h)'); ax7.set_title('Producción de Vapor')
ax7.legend(fontsize='small'); ax7.grid(True, alpha=0.3)

# ax8: Ganancia térmica del campo (real vs simulada — dos métricas)
ax8.plot(res_df['timestamp'], res_df['Gain_Real'],    label='Ganancia Real (Δh local, suav. 5min)', color='tab:red',   linewidth=1.5)
ax8.plot(res_df['timestamp'], res_df['Gain_Sim_Sun'], label='Sim Sun (Q_abs−Q_loss)',               color='gold',      linestyle='--', linewidth=1.5)
ax8.plot(res_df['timestamp'], res_df['Gain_Sim_Ent'], label='Sim Ent (Δh entálpico)',               color='tab:orange',linestyle=':',  linewidth=1.5)
ax8.set_ylabel('Potencia (kW)'); ax8.set_title('Ganancia Térmica Campo Solar')
ax8.legend(fontsize='small'); ax8.grid(True, alpha=0.3)

# ax9: Desglose energético simulado
ax9.plot(res_df['timestamp'], res_df['Q_abs'],      label='Q_abs (absorbida)',  color='gold',       linewidth=1.5)
ax9.plot(res_df['timestamp'], res_df['Q_return'],   label='Q_return (al drum)', color='tab:orange', linestyle='--', linewidth=1.5)
ax9.plot(res_df['timestamp'], res_df['Q_loss_tube'],label='Q_loss (tubo)',      color='tab:gray',   linestyle='--', linewidth=1.0)
ax9.set_ylabel('Potencia (kW)'); ax9.set_title('Desglose Energético (Sim)')
ax9.legend(fontsize='small'); ax9.grid(True, alpha=0.3)

for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig.suptitle(
    f'Digital Twin DSG Caren — {SIM_DATE}  |  Setpoint: {P_SET_G:.1f} bar g',
    fontsize=14, fontweight='bold'
)
fig.savefig(f"tests/validation_{SIM_DATE}.png", dpi=150, bbox_inches='tight')
print(f"Gráfico guardado: tests/validation_{SIM_DATE}.png")
plt.show()
