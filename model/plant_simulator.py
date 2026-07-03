"""
PlantSimulator: capa de abstracción entre el modelo térmico y datos de planta.

Encapsula toda la lógica de control, inicialización de estado y bucle de
simulación, de modo que el script de validación queda reducido a:

    sim    = PlantSimulator(config)
    result = sim.run(df)

El DataFrame devuelto contiene tanto las series simuladas como los datos
reales alineados por timestamp, listos para plotear o exportar.
"""

import numpy as np
import pandas as pd

try:
    from .thermal_model_v2 import ThermalModelV2
    from .pressure_controller import compute_valve_opening
except ImportError:
    from model.thermal_model_v2 import ThermalModelV2
    from model.pressure_controller import compute_valve_opening

CP_W = 4.18   # kJ/(kg·K)


class PlantSimulator:
    """
    Simula la planta DSG sobre un DataFrame de mediciones reales.

    Parámetros de control
    ─────────────────────
    p_setpoint_bar_g  : setpoint de presión manométrica [bar]
    valve_offset_bar  : banda proporcional de la válvula [bar]
    nivel_drum_mm     : setpoint de nivel del drum [mm]
    ghi_threshold     : irradiancia mínima para activar seguimiento solar [W/m²]
    tracking_delay_min: minutos de seguimiento antes de enfocar [min]
    t_feed_c          : temperatura del agua de alimentación [°C]
    m_feed_kgs        : caudal de la bomba de alimentación [kg/s]
    m_rec_nominal_lmin: caudal nominal de recirculación [L/min]
    t_min_pump_c      : temperatura mínima del drum para mantener bomba activa [°C]
                        La bomba sigue corriendo mientras T_drum > este valor,
                        aunque no haya radiación suficiente para seguimiento.
    sub_steps         : sub-pasos por minuto (estabilidad numérica)
    """

    def __init__(self, config: dict,
                 p_setpoint_bar_g: float  = 2.0,
                 valve_offset_bar: float  = 0.5,
                 nivel_drum_mm: float     = 200.0,
                 ghi_threshold: float     = 200.0,
                 tracking_delay_min: int  = 15,
                 t_feed_c: float          = 15.0,
                 m_feed_kgs: float        = 0.01,
                 m_rec_nominal_lmin: float = 100.0,
                 t_min_pump_c: float      = 100.0,
                 sub_steps: int           = 10):
        self.config        = config
        self.model         = ThermalModelV2(config)
        self.P_set         = p_setpoint_bar_g + 1.013   # bar absoluto
        self.valve_offset  = valve_offset_bar
        self.nivel_drum    = nivel_drum_mm
        self.ghi_thresh    = ghi_threshold
        self.track_delay   = tracking_delay_min
        self.t_feed        = t_feed_c
        self.m_feed        = m_feed_kgs
        self.m_rec_nominal = m_rec_nominal_lmin / 60.0   # kg/s
        self.t_min_pump    = t_min_pump_c
        self.sub_steps     = sub_steps

    # ── inicialización de estado ──────────────────────────────────────────────

    def init_state(self, first_row: pd.Series) -> dict:
        """
        Inicializa el estado de simulación a partir del primer punto de datos.

        Mapeo de mediciones a nodos:
          T4  → temperatura del agua fría en el lazo (pre-bomba, bomba, post-bomba, campo)
          T8  → temperatura del drum (retenida del día anterior)
          P5  → presión del drum
          Nivel → nivel de agua del drum
        """
        T_cold  = float(first_row['T4'])
        T_drum  = float(first_row['T8'])
        T_field = float(first_row['T8'])   # T8_sim arranca en T8 real
        level   = float(first_row['Nivel'])
        P_init  = float(first_row['P5']) + 1.013  # bar absoluto

        drum  = self.model.drum
        area  = np.pi * (drum.drum_d / 2) ** 2
        vol   = (level / 1000.0) * area
        props = drum.water_props(P_init, T_C=T_drum)
        M     = vol * props['rho']
        U     = M * props['u'] + drum.m_metal * drum.cp_metal * T_drum

        return {
            "P_drum":      P_init,
            "T_drum":      T_drum,
            "M_total":     M,
            "U_total":     U,
            "level":       level,
            "T_pre_pump":  T_cold,
            "T_pump_bop":  T_cold,
            "T_post_pump": T_cold,
            "T_tube":      T_field,
            "T_field":     T_field,
        }

    # ── ganancia entálpica real ───────────────────────────────────────────────

    def _enthalpy_gain(self, row: pd.Series) -> float:
        """Ganancia térmica real por diferencia de entalpía h(P,T) en T4 y T8.
        Las temperaturas se recortan a T_sat(P) para evitar saltos de fase
        cuando el sensor de presión subestima la presión real."""
        try:
            p4     = float(row.get('P4_real', row.get('P4', np.nan))) + 1.013
            p8     = float(row.get('P8_real', row.get('P8', np.nan))) + 1.013
            t4     = float(row.get('T4_real', row.get('T4', np.nan)))
            t8     = float(row.get('T8_real', row.get('T8', np.nan)))
            caudal = float(row['CAUDAL'])
            if np.isnan(p4) or np.isnan(p8) or caudal == 0:
                return np.nan
            t4_sat = self.model.drum.water_props(p4, x=0)['T']
            t8_sat = self.model.drum.water_props(p8, x=0)['T']
            p4_pr  = self.model.drum.water_props(p4, T_C=min(t4, t4_sat))
            p8_pr  = self.model.drum.water_props(p8, T_C=min(t8, t8_sat))
            m_dot  = (caudal / 1000.0 / 60.0) * p4_pr['rho']
            return m_dot * (p8_pr['h'] - p4_pr['h'])
        except Exception:
            return np.nan

    # ── bucle principal ───────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta la simulación sobre el DataFrame de datos reales.

        El DataFrame de entrada debe tener las columnas:
          timestamp, RAD, T_AMB, T4, T8, T5, T3, P4, P5, P8,
          CAUDAL, EV FB, CAUDAL VAP, Nivel

        Devuelve un DataFrame con todas las series simuladas y reales,
        listo para visualización y análisis.
        """
        state          = self.init_state(df.iloc[0])
        tracking_timer = 0
        feed_active    = False
        results        = []
        dt_sub         = 1.0 / self.sub_steps   # minutos por sub-paso

        for _, row in df.iterrows():
            ghi   = float(row['RAD'])
            t_amb = float(row['T_AMB'])

            # ── lógica de seguimiento solar ───────────────────────────────────
            if ghi > self.ghi_thresh:
                tracking_timer += 1
            else:
                tracking_timer = 0
            focused  = tracking_timer >= self.track_delay
            dni_step = ghi if focused else 0.0

            # ── control de bomba de recirculación ────────────────────────────
            # La bomba corre si hay radiación suficiente O si el drum sigue
            # caliente (T_drum > t_min_pump). Esto replica el comportamiento
            # real: al atardecer la bomba sigue activa mientras hay calor útil.
            solar_active = ghi > self.ghi_thresh
            drum_hot     = state['T_drum'] > self.t_min_pump
            pump_active  = solar_active or drum_hot

            if pump_active:
                if state['level'] < self.nivel_drum:
                    feed_active = True
                elif state['level'] > self.nivel_drum + 50:
                    feed_active = False
                m_feed = self.m_feed if feed_active else 0.0
                m_rec  = self.m_rec_nominal
            else:
                feed_active = False
                m_feed      = 0.0
                m_rec       = 0.0

            inputs = {
                "timestamp":  row['timestamp'],
                "dni":        dni_step,
                "t_amb":      t_amb,
                "m_rec":      m_rec,
                "m_feed":     m_feed,
                "valve_open": 0.0,   # se actualiza en cada sub-paso
                "eta_opt":    self.config['collector']['optical_efficiency_0'],
                "T_feed":     self.t_feed,
                "debug":      False,
            }

            # ── sub-pasos (estabilidad numérica) ──────────────────────────────
            for _ in range(self.sub_steps):
                inputs['valve_open'] = compute_valve_opening(
                    state['P_drum'], self.P_set, self.valve_offset)
                out = self.model.simulate_step(state, inputs, dt_min=dt_sub)

                state = {
                    "P_drum":      max(1.013, min(10.0, out['P_drum_sim'])),
                    "T_drum":      out['T_drum_sim'],
                    "M_total":     out['M_total_new'],
                    "U_total":     out['U_total_new'],
                    "level":       out['level_sim'],
                    "T_pre_pump":  out['T_pre_pump_sim'],
                    "T_pump_bop":  out['T_pump_bop_sim'],
                    "T_post_pump": out['T_post_pump_sim'],
                    "T_tube":      out['T_tube_sim'],
                    "T_field":     out['T_field_sim'],
                    "h_field":     out['h_field_sim'],
                    "x_field":     out['x_field_sim'],
                    "rho_field":   out['rho_field_sim'],
                }

            valve_pos = compute_valve_opening(state['P_drum'], self.P_set, self.valve_offset)

            results.append({
                # Tiempo
                "timestamp":   row['timestamp'],
                # Simulación — drum
                "P_sim":       state['P_drum'] - 1.013,
                "T_drum_sim":  state['T_drum'],
                "Level_sim":   state['level'],
                "Valve_sim":   valve_pos,
                "Vapor_sim":   out['m_vapor_sim'] * 3600,   # kg/h
                # Simulación — nodos de tubería
                "T_pre_pump_sim": state['T_pre_pump'],
                "T_pump_bop_sim": state['T_pump_bop'],
                "T4_sim":         state['T_post_pump'],      # ≈ T4
                # Simulación — campo solar
                "T_tube_sim":     state['T_tube'],
                "T8_sim":         state['T_field'],          # ≈ T8
                "X_field_sim":    out['x_field_sim'],         # calidad de vapor [0-1]
                # Energía
                "Q_abs":          out['Q_abs'],
                "Q_return":       out['Q_return'],
                "Q_gain_sun":     out['Q_gain_sun'],
                "Q_gain_ent":     out['Q_gain_ent'],
                "Q_loss_tube":    out['Q_loss_tube'],
                "DNI_est":        out['dni_est'],
                "eta_total":      out['eta_total'],
                # Caudal simulado (valor fijo nominal cuando bomba activa)
                "CAUDAL_sim":     m_rec * 60.0,        # L/min
                "pump_active_sim": int(pump_active),    # 1=activa, 0=detenida
                # Datos reales
                "RAD":            ghi,
                "T_AMB":          t_amb,
                "P5_real":        float(row['P5']),
                "T5_real":        float(row['T5']),
                "T8_real":        float(row['T8']),
                "T4_real":        float(row['T4']),
                "T3_real":        float(row['T3']),
                "CAUDAL":         float(row['CAUDAL']),
                "Valve_real":     float(row['EV FB']),
                "Vapor_real":     float(row['CAUDAL VAP']),
                "Level_real":     float(row['Nivel']),
                "P4_real":        float(row.get('P4', np.nan)),
                "P8_real":        float(row.get('P8', np.nan)),
            })

        res_df = pd.DataFrame(results)

        # ── ganancia energética real ──────────────────────────────────────────
        res_df['Gain_Real_Cp'] = (
            (res_df['CAUDAL'] / 60.0) * CP_W
            * (res_df['T8_real'] - res_df['T4_real'])
        )
        res_df['Gain_Real_h'] = res_df.apply(self._enthalpy_gain, axis=1)
        res_df['Gain_Real']     = (
            res_df['Gain_Real_h'].fillna(res_df['Gain_Real_Cp']).clip(lower=0)
        )
        res_df.loc[res_df['Gain_Real'] > 50, 'Gain_Real'] = 0
        # Suavizado: ventana móvil de 5 min para atenuar ruido de sensores
        res_df['Gain_Real'] = (
            res_df['Gain_Real'].rolling(window=5, center=True, min_periods=1).mean()
        )
        res_df['Gain_Sim_Sun'] = res_df['Q_gain_sun']
        res_df['Gain_Sim_Ent'] = res_df['Q_gain_ent']

        return res_df
