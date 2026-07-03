"""
Thermal model v2 — arquitectura explícita de nodos.

Topología del circuito de recirculación:

  [Steam Drum] ──► [pipe_pre_pump] ──► [pump_bop] ──► [pipe_post_pump]
       ▲                                                       │
       │                                                       ▼
       └─────────────────── [SolarFieldNode] ◄────────────────┘
                              (T_tube + T_fluid)

Mediciones reales que se mapean a los nodos:
  T5  →  T_drum     (temperatura/presión en el steam drum)
  T4  →  T_post_pump (salida del nodo post-bomba, entrada al campo)
  T8  →  T_field    (salida del campo solar)

Clase base ThermalPipeNode
──────────────────────────
Balance de energía implícito (Euler), incondicionalmente estable:

  C/dt·(T_new − T_prev) = ṁ·cp·(T_in − T_new) − UA·(T_new − T_amb)

  C   = m_fluido·cp_w + m_metal·cp_metal   [kJ/K]
  ṁ   = caudal másico                       [kg/s]
  UA  = coef. pérdida × área               [kW/K]

SolarFieldNode añade dos estados acoplados (tubo metálico + fluido) y las
ecuaciones de ganancia/pérdida solar del receptor evacuado.
"""

import numpy as np
from iapws import IAPWS97

try:
    import pandas as pd
    from pvlib.solarposition import get_solarposition
    from utils.solar_utils import estimate_dni_perez
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False

CP_W   = 4.18    # kJ/(kg·K)  agua líquida
RHO_W  = 990.0   # kg/m³      agua líquida caliente (aprox)


def rho_liquid(T_C: float) -> float:
    """Densidad del agua líquida [kg/m³], válida 0–150 °C (polinomio ajustado IAPWS)."""
    T = float(np.clip(T_C, 0.0, 150.0))
    return 1000.15 - 0.0559*T - 0.00363*T**2 + 1.56e-6*T**3


# ─────────────────────────────────────────────────────────────────────────────
# Clase base: nodo de tubería / componente de lazo
# ─────────────────────────────────────────────────────────────────────────────

class ThermalPipeNode:
    """
    Nodo genérico de fase líquida: volumen de fluido + masa metálica + pérdidas al ambiente.

    Un mismo objeto modela la cañería pre-bomba, la bomba/BoP y la cañería post-bomba;
    sólo difieren en sus parámetros físicos.
    """

    def __init__(self, fluid_volume_L: float, metal_mass_kg: float,
                 metal_cp_kj_kgk: float, ua_loss_kw_k: float, name: str = "pipe"):
        self.V_fluid  = fluid_volume_L / 1000.0  # m³
        self.m_metal  = metal_mass_kg
        self.cp_metal = metal_cp_kj_kgk          # kJ/(kg·K)
        self.UA_loss  = ua_loss_kw_k             # kW/K
        self.name     = name

    def capacity(self, rho: float = RHO_W, cp_w: float = CP_W) -> float:
        """Capacidad térmica efectiva [kJ/K]."""
        return self.V_fluid * rho * cp_w + self.m_metal * self.cp_metal

    def step(self, T_prev: float, T_in: float, m_dot: float, T_amb: float,
             dt_s: float, cp_w: float = CP_W, rho: float = None) -> tuple:
        """
        Avanza un paso de tiempo.

        T_prev : temperatura actual del nodo [°C]
        T_in   : temperatura del fluido entrante (nodo anterior) [°C]
        m_dot  : caudal másico [kg/s]  (0 si la bomba está apagada)
        T_amb  : temperatura ambiente [°C]
        dt_s   : paso de tiempo [s]
        Retorna (T_new [°C], M_expelled [kg]).
        M_expelled > 0 si el fluido se expande (calentamiento); < 0 si contrae.
        """
        rho_prev = rho if rho is not None else rho_liquid(T_prev)
        C    = self.capacity(rho_prev, cp_w)
        m_cp = m_dot * cp_w                  # acoplamiento advectivo [kW/K]

        denom = C / dt_s + self.UA_loss + m_cp
        numer = (C / dt_s) * T_prev + m_cp * T_in + self.UA_loss * T_amb
        T_new = numer / denom

        M_expelled = self.V_fluid * (rho_prev - rho_liquid(T_new))
        return T_new, M_expelled


# ─────────────────────────────────────────────────────────────────────────────
# Nodo campo solar
# ─────────────────────────────────────────────────────────────────────────────

class SolarFieldNode:
    """
    Campo solar Fresnel con dos estados acoplados:
      T_tube  — temperatura del metal del tubo receptor absorbedor
      T_fluid — temperatura del fluido en el campo (bien mezclado)

    Flujos de calor:
      Q_abs  [kW]  →  tubo (absorción solar)
      Q_loss [kW]  ←  tubo → ambiente (pérdidas del receptor evacuado)
      Q_conv [kW]  →  fluido desde el tubo (convección forzada h·A)

    T_in  (entrada campo ≈ T4)  →  T_fluid_new  (salida campo ≈ T8)
    """

    def __init__(self, tube_mass_kg: float, tube_cp_kj_kgk: float,
                 fluid_volume_L: float, aperture_area_m2: float,
                 thermal_loss_coeffs: list, ua_tube_fluid_kw_k: float):
        self.m_tube        = tube_mass_kg
        self.cp_tube       = tube_cp_kj_kgk
        self.V_fluid       = fluid_volume_L / 1000.0
        self.area          = aperture_area_m2
        self.a1, self.a2   = thermal_loss_coeffs
        self.UA_tube_fluid = ua_tube_fluid_kw_k   # kW/K  tubo→fluido

    def q_abs(self, dni: float, eta_opt: float, iam: float,
              optical_loss: float) -> float:
        """Potencia absorbida por el tubo [kW]."""
        return dni * self.area * eta_opt * optical_loss * iam / 1000.0

    def q_loss(self, T_tube: float, T_amb: float) -> float:
        """Pérdidas del tubo receptor al ambiente [kW]."""
        dT = T_tube - T_amb
        return self.area * (self.a1 * dT + self.a2 * dT ** 2) / 1000.0

    @staticmethod
    def fluid_state(P_bar: float, h_kj_kg: float) -> tuple:
        """(T_C, x, rho) del fluido a partir de entalpía y presión (flash P,h)."""
        s = IAPWS97(P=P_bar / 10.0, h=h_kj_kg)
        return s.T - 273.15, s.x, s.rho

    def step(self, T_tube_prev: float, h_fluid_prev: float, T_fluid_prev: float,
             x_fluid_prev: float, rho_fluid_prev: float,
             h_in: float, m_dot: float, T_amb: float,
             dt_s: float, Q_abs: float, P_ref: float) -> tuple:
        """
        Avanza un paso de tiempo para el campo solar.

        El fluido se rastrea por entalpía específica (no temperatura), de modo
        que al cruzar la curva de saturación la temperatura queda naturalmente
        fija en T_sat(P_ref) mientras la calidad de vapor x crece — sin recortes
        artificiales ni pérdida de la energía latente absorbida.

        T_tube_prev  : temperatura del metal del receptor [°C]
        h_fluid_prev, T_fluid_prev, x_fluid_prev, rho_fluid_prev :
                       estado termodinámico del fluido en el paso anterior
                       (evita repetir el flash IAPWS de (P,h) dos veces por paso)
        h_in         : entalpía de entrada al campo (≈ T4) [kJ/kg]
        m_dot        : caudal másico [kg/s]
        T_amb        : temperatura ambiente [°C]
        Q_abs        : energía solar absorbida por el tubo [kW]
        P_ref        : presión de referencia del campo [bar]
        Retorna (T_tube_new, h_fluid_new, T_fluid_new, x_fluid_new, rho_fluid_new, M_expelled [kg]).
        T_fluid_new ≈ T8 (salida del campo); x_fluid_new = calidad de vapor a la salida.
        """
        C_tube  = self.m_tube * self.cp_tube
        M_fluid = self.V_fluid * rho_fluid_prev

        Q_loss_t = self.q_loss(T_tube_prev, T_amb)

        # Tubo: implícito en UA_tube_fluid
        # C_tube/dt*(T_tube_new - T_tube) = Q_abs - Q_loss - UA_tf*(T_tube_new - T_fluid_prev)
        UA_tf       = self.UA_tube_fluid if m_dot > 0 else 0.0
        denom_tube  = C_tube / dt_s + UA_tf
        numer_tube  = (C_tube / dt_s) * T_tube_prev + Q_abs - Q_loss_t + UA_tf * T_fluid_prev
        T_tube_new  = numer_tube / denom_tube

        # Calor convectivo real transferido al fluido con la T_tube ya actualizada
        Q_conv = UA_tf * (T_tube_new - T_fluid_prev)

        # Fluido: balance de entalpía, implícito en advección
        # M_fluid/dt*(h_new - h_prev) = Q_conv + m_dot*(h_in - h_new)
        denom_fluid  = M_fluid / dt_s + m_dot
        numer_fluid  = (M_fluid / dt_s) * h_fluid_prev + Q_conv + m_dot * h_in
        h_fluid_new  = numer_fluid / denom_fluid

        T_fluid_new, x_fluid_new, rho_fluid_new = self.fluid_state(P_ref, h_fluid_new)

        # Expansión térmica: sólo tiene sentido en fase líquida pura. Si hay
        # vapor formándose, el exceso de volumen se transporta como calidad
        # dentro del propio flujo (ya capturado por h_fluid_new), no como
        # "expulsión" por densidad.
        if x_fluid_prev <= 0.0 and x_fluid_new <= 0.0:
            M_expelled = self.V_fluid * (rho_fluid_prev - rho_fluid_new)
        else:
            M_expelled = 0.0

        return T_tube_new, h_fluid_new, T_fluid_new, x_fluid_new, rho_fluid_new, M_expelled


# ─────────────────────────────────────────────────────────────────────────────
# Nodo steam drum
# ─────────────────────────────────────────────────────────────────────────────

class SteamDrumNode:
    """
    Steam drum bifásico con termodinámica IAPWS-IF97.
    Balance de masa y energía interna; la presión y temperatura se derivan
    del equilibrio termodinámico en el volumen fijo del drum.
    """

    def __init__(self, diameter_mm: float, height_mm: float,
                 metal_mass_kg: float, heat_loss_coeff_w_m2k: float,
                 metal_cp_kj_kgk: float = 0.5):
        self.drum_d    = diameter_mm / 1000.0
        self.drum_h    = height_mm  / 1000.0
        self.drum_vol  = np.pi * (self.drum_d / 2) ** 2 * self.drum_h
        self.m_metal   = metal_mass_kg
        self.h_loss    = heat_loss_coeff_w_m2k / 1000.0   # kW/(m²·K)
        self.cp_metal  = metal_cp_kj_kgk
        # Área superficial: manto cilíndrico + 2 tapas
        self.area = (np.pi * self.drum_d * self.drum_h
                     + 2 * np.pi * (self.drum_d / 2) ** 2)

    # ── propiedades termodinámicas ───────────────────────────────────────────

    def water_props(self, P_bar: float, T_C: float = None, x: float = None) -> dict:
        P_mpa = P_bar / 10.0
        if x is not None:
            s = IAPWS97(P=P_mpa, x=x)
        elif T_C is not None:
            s = IAPWS97(P=P_mpa, T=T_C + 273.15)
        else:
            s = IAPWS97(P=P_mpa, x=0)
        return {"h": s.h, "rho": s.rho, "u": s.u, "T": s.T - 273.15}

    # ── nivel ────────────────────────────────────────────────────────────────

    def level_mm(self, M_liq: float, P_bar: float) -> float:
        rho_f = self.water_props(P_bar, x=0)['rho']
        area  = np.pi * (self.drum_d / 2) ** 2
        return (M_liq / rho_f / area) * 1000.0

    # ── paso de tiempo ───────────────────────────────────────────────────────

    def step(self, M_total: float, U_total: float, P_prev: float,
             m_feed: float, h_feed: float,
             m_vapor: float, h_vap: float,
             Q_return: float, T_amb: float, dt_s: float,
             m_expelled: float = 0.0, h_expelled: float = 0.0) -> dict:
        """
        Avanza el drum un paso de tiempo.

        Q_return   [kW]   : calor aportado por el retorno caliente del campo solar.
        m_expelled [kg/s] : flujo másico por expansión térmica del circuito (>0 entra al drum).
        h_expelled [kJ/kg]: entalpía del fluido expelido desde el circuito.
        Retorna dict con el nuevo estado termodinámico completo.
        """
        T_sat   = self.water_props(P_prev, x=0)['T']
        Q_loss  = self.h_loss * self.area * (T_sat - T_amb)   # kW

        M_new = M_total + (m_feed - m_vapor + m_expelled) * dt_s
        U_new = U_total + (m_feed * h_feed
                           - m_vapor * h_vap
                           + m_expelled * h_expelled
                           + Q_return
                           - Q_loss) * dt_s

        P_new, T_new, quality = self._equilibrium(M_new, U_new, P_prev)
        P_new = float(np.clip(P_new, 1.013, 10.0))

        M_liq = M_new * (1.0 - quality)
        if quality == 0.0:
            # Subenfriado: la densidad real difiere de rho_sat; usarla evita un
            # salto espurio de nivel al arranque cuando T_liq << T_sat(P).
            rho_sc = self.water_props(P_new, T_C=T_new)['rho']
            area   = np.pi * (self.drum_d / 2) ** 2
            level  = (M_liq / rho_sc / area) * 1000.0
        else:
            level  = self.level_mm(M_liq, P_new)

        return {
            "M_total": M_new,
            "U_total": U_new,
            "P":       P_new,
            "T":       T_new,
            "quality": quality,
            "level":   level,
            "Q_loss":  Q_loss,
        }

    # ── solucionador de equilibrio termodinámico (igual al v1) ───────────────

    def _equilibrium(self, M: float, U: float, P_hint: float = None,
                     max_iter: int = 10) -> tuple:
        P_bar = max(1.013, P_hint or 1.013)

        # Verificar si está subenfriado
        try:
            sf = self.water_props(P_bar, x=0)
            U_sat = M * sf['u'] + self.m_metal * self.cp_metal * sf['T']
            if U < U_sat:
                T_lo, T_hi = 0.01, max(0.01, sf['T'] - 0.01)
                for _ in range(30):
                    T_m = 0.5 * (T_lo + T_hi)
                    sl  = self.water_props(P_bar, T_C=T_m)
                    U_m = M * sl['u'] + self.m_metal * self.cp_metal * T_m
                    if U_m < U:
                        T_lo = T_m
                    else:
                        T_hi = T_m
                return P_bar, 0.5 * (T_lo + T_hi), 0.0
        except Exception:
            pass

        # Zona bifásica — iteraciones Newton
        from iapws.iapws97 import _PSat_T
        T_g = 120.0
        for _ in range(max_iter):
            T_g = float(np.clip(T_g, 0.01, 373.0))
            try:
                P_bar = _PSat_T(T_g + 273.15) * 10.0
            except Exception:
                P_bar = max(1.013, (T_g / 100.0) ** 4)

            sf = self.water_props(P_bar, x=0)
            sg = self.water_props(P_bar, x=1)
            v_f, v_g = 1.0 / sf['rho'], 1.0 / sg['rho']

            v_avg   = self.drum_vol / M
            quality = float(np.clip((v_avg - v_f) / (v_g - v_f), 0.0, 1.0))
            u_water = sf['u'] + quality * (sg['u'] - sf['u'])
            U_calc  = M * u_water + self.m_metal * self.cp_metal * T_g
            err     = U - U_calc

            # Derivada numérica dU/dT
            dT = 0.1
            try:
                Pp = _PSat_T(T_g + dT + 273.15) * 10.0
            except Exception:
                Pp = max(1.013, ((T_g + dT) / 100.0) ** 4)
            sfp = self.water_props(Pp, x=0)
            sgp = self.water_props(Pp, x=1)
            v_fp, v_gp = 1.0 / sfp['rho'], 1.0 / sgp['rho']
            qp  = float(np.clip((v_avg - v_fp) / (v_gp - v_fp), 0.0, 1.0))
            U_p = M * (sfp['u'] + qp * (sgp['u'] - sfp['u'])) + self.m_metal * self.cp_metal * (T_g + dT)
            dU_dT = (U_p - U_calc) / dT

            T_g += err / dU_dT if abs(dU_dT) > 0.01 else err / (M * CP_W + self.m_metal * self.cp_metal)
            if abs(err) < 1.0:
                break

        return max(1.013, P_bar), T_g, quality


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador principal
# ─────────────────────────────────────────────────────────────────────────────

class ThermalModelV2:
    """
    Modelo termico v2: red explícita de nodos en el lazo de recirculación.

    Secciones requeridas en plant_config.json (además de las existentes):
      "solar_field"    : {tube_mass_kg, tube_cp, fluid_volume_L}
      "pipe_pre_pump"  : {fluid_volume_L, metal_mass_kg, metal_cp, ua_loss_kw_k}
      "pump_bop"       : {fluid_volume_L, metal_mass_kg, metal_cp, ua_loss_kw_k}
      "pipe_post_pump" : {fluid_volume_L, metal_mass_kg, metal_cp, ua_loss_kw_k}

    Estado (state dict):
      P_drum, M_total, U_total  — termodinámica del drum
      T_drum                    — temperatura del drum (se actualiza cada paso)
      T_pre_pump                — temperatura cañería drum→bomba
      T_pump_bop                — temperatura nodo bomba/válvulas/BoP
      T_post_pump               — temperatura cañería bomba→campo  (≈ T4)
      T_tube                    — temperatura metal receptor solar
      T_field                   — temperatura fluido en campo            (≈ T8)
      level                     — nivel steam drum [mm]
    """

    def __init__(self, config: dict):
        self.config = config
        loc = config['location']
        self.lat = loc['latitude']
        self.lon = loc['longitude']
        self.alt = loc['altitude']
        self.tz  = loc.get('timezone', 'UTC')

        # Steam drum
        sd = config['steam_drum']
        self.drum = SteamDrumNode(
            diameter_mm           = sd['diameter_mm'],
            height_mm             = sd['height_mm'],
            metal_mass_kg         = sd.get('metal_mass_kg', 120),
            heat_loss_coeff_w_m2k = sd.get('heat_loss_coeff_w_m2k', 2.0),
        )

        # Nodos de tubería (en orden de flujo)
        self.pre_pump  = self._pipe_node(config['pipe_pre_pump'],  'pre_pump')
        self.pump_bop  = self._pipe_node(config['pump_bop'],       'pump_bop')
        self.post_pump = self._pipe_node(config['pipe_post_pump'], 'post_pump')

        # Campo solar
        coll     = config['collector']
        sf       = config['solar_field']
        tube_len = sf.get('tube_length_m', 21.0)
        D_int    = sf.get('tube_inner_diameter_mm', 40.0) / 1000.0   # m
        h_tf     = sf.get('h_tube_fluid_w_m2k', 1500.0)
        UA_tf    = h_tf * np.pi * D_int * tube_len / 1000.0          # kW/K

        self.field = SolarFieldNode(
            tube_mass_kg        = sf['tube_mass_kg'],
            tube_cp_kj_kgk      = sf.get('tube_cp', 0.5),
            fluid_volume_L      = sf['fluid_volume_L'],
            aperture_area_m2    = coll['aperture_area'],
            thermal_loss_coeffs = coll['thermal_loss_coefficients'],
            ua_tube_fluid_kw_k  = UA_tf,
        )

        self.eta0           = coll['optical_efficiency_0']
        self.optical_loss   = coll.get('optical_loss_factor', 1.0)
        self.iam_cfg        = coll.get('iam', {})
        self.kv             = config.get('steam_valve', {}).get('kv', 0.012)
        self.dP_pump_bar    = config.get('hydraulics', {}).get('delta_P_pump_bar', 0.54)

    # ── constructores de nodos ────────────────────────────────────────────────

    @staticmethod
    def _pipe_node(cfg: dict, name: str) -> ThermalPipeNode:
        return ThermalPipeNode(
            fluid_volume_L  = cfg.get('fluid_volume_L', 5.0),
            metal_mass_kg   = cfg.get('metal_mass_kg', 30.0),
            metal_cp_kj_kgk = cfg.get('metal_cp', 0.5),
            ua_loss_kw_k    = cfg.get('ua_loss_kw_k', 0.03),
            name            = name,
        )

    # ── IAM ──────────────────────────────────────────────────────────────────

    def _iam(self, theta_l: float, theta_t: float) -> float:
        if not self.iam_cfg:
            return 1.0
        angles = self.iam_cfg['angles']
        iam_l  = np.interp(abs(theta_l), angles, self.iam_cfg['longitudinal'])
        iam_t  = np.interp(abs(theta_t), angles, self.iam_cfg['transversal'])
        return iam_l * iam_t

    # ── cálculo solar ─────────────────────────────────────────────────────────

    def _solar(self, ghi: float, timestamp, eta_opt: float) -> tuple:
        """Retorna (Q_abs [kW], dni_efectivo, iam_factor, eta_total)."""
        iam_factor = 1.0
        dni        = ghi

        if PVLIB_AVAILABLE and timestamp is not None and ghi > 0:
            try:
                ts = pd.Timestamp(timestamp)
                if ts.tz is None:
                    ts = ts.tz_localize(self.tz)

                sol     = get_solarposition(ts, self.lat, self.lon, self.alt)
                zen_deg = sol['zenith'].values[0]
                az_rad  = np.radians(sol['azimuth'].values[0])
                zen_rad = np.radians(zen_deg)

                dni = estimate_dni_perez(ghi, zen_deg, ts)

                theta_l    = np.degrees(np.arcsin(np.sin(zen_rad) * np.sin(az_rad)))
                theta_t    = np.degrees(np.arctan2(
                    np.sin(zen_rad) * np.cos(az_rad), np.cos(zen_rad)))
                iam_factor = self._iam(theta_l, theta_t)
            except Exception:
                pass

        Q_abs = self.field.q_abs(dni, eta_opt, iam_factor, self.optical_loss)
        return Q_abs, dni, iam_factor, eta_opt * iam_factor

    # ── paso principal ────────────────────────────────────────────────────────

    def simulate_step(self, state: dict, inputs: dict, dt_min: float) -> dict:
        """
        Avanza toda la planta un paso de tiempo.

        Entradas en `inputs`:
          timestamp, dni (GHI medido), t_amb, m_rec [kg/s], m_feed [kg/s],
          valve_open [%], eta_opt, T_feed [°C], debug [bool]
        """
        dt    = dt_min * 60.0
        t_amb = inputs['t_amb']
        m_rec = inputs.get('m_rec', 0.0)

        # Desempacar estado
        P_drum  = state['P_drum']
        M_total = state['M_total']
        U_total = state['U_total']
        T_drum  = state.get('T_drum', 25.0)

        T_pre   = state.get('T_pre_pump',  T_drum)
        T_bop   = state.get('T_pump_bop',  T_drum)
        T_post  = state.get('T_post_pump', T_drum)
        T_tube  = state.get('T_tube',      T_drum)

        # ── 1. Solar ──────────────────────────────────────────────────────────
        eta_opt = inputs.get('eta_opt', self.eta0)
        Q_abs, dni_eff, iam, eta_total = self._solar(
            inputs.get('dni', 0.0), inputs.get('timestamp'), eta_opt)

        # ── 2. Nodos de tubería: drum → campo ─────────────────────────────────
        # Sin flujo (m_rec=0): sólo se enfrían al ambiente.
        # Con flujo: advección desde el nodo anterior.
        T_pre_new,  M_exp_pre   = self.pre_pump.step( T_pre,  T_drum,    m_rec, t_amb, dt)
        T_bop_new,  M_exp_bop   = self.pump_bop.step( T_bop,  T_pre_new, m_rec, t_amb, dt)
        T_post_new, M_exp_post  = self.post_pump.step(T_post, T_bop_new, m_rec, t_amb, dt)

        # ── 3. Campo solar (fluido rastreado por entalpía) ────────────────────
        # P_field_ref: presión estimada aguas abajo de la bomba (P_drum + ΔP_bomba).
        # El campo es el único nodo donde el fluido puede alcanzar saturación en
        # operación real, por eso su estado es (h, x) en vez de sólo T.
        P_field_ref = P_drum + self.dP_pump_bar
        T_sat_field = self.drum.water_props(P_field_ref, x=0)['T']

        h_field_prev = state.get('h_field')
        if h_field_prev is None:
            T_field_prev0  = state.get('T_field', T_drum)
            h_field_prev   = self.drum.water_props(
                P_field_ref, T_C=min(T_field_prev0, T_sat_field - 0.01))['h']
            T_field_prev, x_field_prev, rho_field_prev = self.field.fluid_state(
                P_field_ref, h_field_prev)
        else:
            T_field_prev  = state['T_field']
            x_field_prev  = state.get('x_field', 0.0)
            rho_field_prev = state.get('rho_field', rho_liquid(T_field_prev))

        h_field_in = self.drum.water_props(
            P_field_ref, T_C=min(T_post_new, T_sat_field - 0.01))['h']

        (T_tube_new, h_field_new, T_field_new, x_field_new, rho_field_new,
         M_exp_field) = self.field.step(
            T_tube_prev      = T_tube,
            h_fluid_prev     = h_field_prev,
            T_fluid_prev     = T_field_prev,
            x_fluid_prev     = x_field_prev,
            rho_fluid_prev   = rho_field_prev,
            h_in             = h_field_in,
            m_dot            = m_rec,
            T_amb            = t_amb,
            dt_s             = dt,
            Q_abs            = Q_abs,
            P_ref            = P_field_ref,
        )

        # ── 3b. Expansión térmica del circuito ────────────────────────────────
        # Masa total expelida desde los nodos hacia el drum en este paso [kg].
        # Se convierte a caudal [kg/s] para el balance de masa del drum.
        M_exp_total = M_exp_pre + M_exp_bop + M_exp_post + M_exp_field
        m_exp_rate  = M_exp_total / dt

        # Entalpía del fluido expelido: promedio ponderado de los nodos que expanden.
        # Si hay contracción (enfriamiento), el drum provee fluido líquido saturado.
        T_sat_drum = self.drum.water_props(P_drum, x=0)['T']
        if M_exp_total > 1e-9:
            def _h_node(T_node):
                return self.drum.water_props(P_drum, T_C=min(T_node, T_sat_drum))['h']
            weights = [max(0.0, M_exp_pre), max(0.0, M_exp_bop),
                       max(0.0, M_exp_post), max(0.0, M_exp_field)]
            temps   = [T_pre_new, T_bop_new, T_post_new, T_field_new]
            w_sum   = sum(weights) + 1e-12
            h_exp   = sum(w * _h_node(T) for w, T in zip(weights, temps)) / w_sum
        elif M_exp_total < -1e-9:
            h_exp = self.drum.water_props(P_drum, T_C=min(T_drum, T_sat_drum))['h']
        else:
            h_exp = 0.0

        # ── 4. Calor retornado al drum y ganancia del campo ───────────────────
        # Q_return: calor neto entregado al drum (balance de masa/energía).
        # Usa entalpías reales (no ΔT×cp) para que el calor latente generado en
        # el campo (si x_field_new > 0) se transporte correctamente hacia el drum.
        h_drum_liquid = self.drum.water_props(P_drum, T_C=min(T_drum, T_sat_drum))['h']
        Q_return = m_rec * (h_field_new - h_drum_liquid) if m_rec > 0 else 0.0

        # Q_loss_tube: pérdida del tubo absorbedor al ambiente [kW]
        Q_loss_tube = self.field.q_loss(T_tube_new, t_amb)

        # Opción 1 — Gain_Sim_Sun: balance energético del campo (no depende de presión)
        # Energía solar absorbida menos pérdidas del tubo; captura calor sensible y latente.
        Q_gain_sun = Q_abs - Q_loss_tube

        # Opción 2 — Gain_Sim_Ent: Δh entre salida y entrada del campo, evaluadas
        # ambas a P_field_ref = P_drum + ΔP_bomba. Al estar el campo en entalpía,
        # Δh ya no requiere recorte a T_sat: si se forma vapor, h_field_new sigue
        # creciendo con la calidad y Δh no colapsa.
        Q_gain_ent = m_rec * (h_field_new - h_field_in) if m_rec > 0 else 0.0

        # ── 5. Válvula y alimentación ─────────────────────────────────────────
        m_vapor = self.kv * (inputs.get('valve_open', 0.0) / 100.0) * np.sqrt(
            max(0.0, P_drum - 1.013))

        h_vap  = self.drum.water_props(P_drum, x=1)['h']
        h_feed = self.drum.water_props(P_drum, T_C=inputs.get('T_feed', 15.0))['h']

        # ── 6. Steam drum ─────────────────────────────────────────────────────
        drum_out = self.drum.step(
            M_total    = M_total,
            U_total    = U_total,
            P_prev     = P_drum,
            m_feed     = inputs.get('m_feed', 0.0),
            h_feed     = h_feed,
            m_vapor    = m_vapor,
            h_vap      = h_vap,
            Q_return   = Q_return,
            T_amb      = t_amb,
            dt_s       = dt,
            m_expelled = m_exp_rate,
            h_expelled = h_exp,
        )

        if inputs.get('debug'):
            print(
                f"DEBUG v2 | Q_abs={Q_abs:.1f}kW  Q_ret={Q_return:.1f}kW "
                f"m_v={m_vapor:.4f}kg/s  P={drum_out['P']:.2f}bar  "
                f"x={drum_out['quality']:.4f} | "
                f"T_pre={T_pre_new:.1f} T_bop={T_bop_new:.1f} "
                f"T4≈{T_post_new:.1f} T8≈{T_field_new:.1f} T_drum={drum_out['T']:.1f}°C"
            )

        return {
            # Steam drum
            "P_drum_sim":    drum_out['P'],
            "T_drum_sim":    drum_out['T'],
            "M_total_new":   drum_out['M_total'],
            "U_total_new":   drum_out['U_total'],
            "level_sim":     drum_out['level'],
            "m_vapor_sim":   m_vapor,
            "Q_loss_drum":   drum_out['Q_loss'],
            "M_exp_total":   M_exp_total,   # kg expelidos por expansión térmica en este paso
            # Nodos tubería
            "T_pre_pump_sim":  T_pre_new,
            "T_pump_bop_sim":  T_bop_new,
            "T_post_pump_sim": T_post_new,   # ≈ T4
            # Campo solar
            "T_tube_sim":    T_tube_new,
            "T_field_sim":   T_field_new,    # ≈ T8
            "h_field_sim":   h_field_new,    # entalpía fluido salida campo [kJ/kg]
            "x_field_sim":   x_field_new,    # calidad de vapor salida campo [0-1]
            "rho_field_sim": rho_field_new,
            # Energía
            "Q_abs":         Q_abs,
            "Q_return":      Q_return,
            "Q_gain_sun":    Q_gain_sun,
            "Q_gain_ent":    Q_gain_ent,
            "Q_loss_tube":   Q_loss_tube,
            "dni_est":       dni_eff,
            "iam":           iam,
            "eta_total":     eta_total,
            # Alias de compatibilidad con real_data_test.py
            "T_sf_out_sim":       T_field_new,
            "T_pipe_sim":         T_post_new,
            "T_loop_sim":         T_post_new,
            "warmup_factor":      1.0,
            "Q_net":              Q_return,
            "Q_abs_effective":    Q_abs,
            "Q_battery_charge":   0.0,
            "Q_battery_fluid":    0.0,
            "Q_battery_loss":     0.0,
        }
