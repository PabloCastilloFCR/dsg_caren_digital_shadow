import numpy as np
from iapws import IAPWS97

try:
    import pandas as pd
    from pvlib.solarposition import get_solarposition
    from utils.solar_utils import estimate_dni_perez
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False

class ThermalModel:
    def __init__(self, config):
        self.config = config
        self.area = config['collector']['aperture_area']
        self.eta0 = config['collector']['optical_efficiency_0']
        
        # Geometría del Drum
        self.drum_d = config['steam_drum']['diameter_mm'] / 1000.0
        self.drum_h = config['steam_drum']['height_mm'] / 1000.0
        self.drum_vol = np.pi * (self.drum_d/2)**2 * self.drum_h
        self.drum_metal_mass = config['steam_drum'].get('metal_mass_kg', 0)
        self.drum_h_loss = config['steam_drum'].get('heat_loss_coeff_w_m2k', 1.0)
        # Área superficial Drum: Manto + Tapas
        self.drum_area = (np.pi * self.drum_d * self.drum_h) + (2 * np.pi * (self.drum_d/2)**2)
        
        # Parámetros de tubería (Inercia)
        self.pipe_vol = config['piping']['fluid_volume_L'] / 1000.0
        self.pipe_metal_mass = config['piping']['total_metal_mass_kg']
        self.pipe_cp = config['piping']['material_cp']
        self.pipe_h_loss = config['piping'].get('heat_loss_coeff_w_m2k', 1.0)
        # Área superficial aprox: pi * D * L (1.5" ~ 0.048m ext)
        self.pipe_area = np.pi * 0.048 * config['piping']['length_m']

        # Calor específico del agua líquida (kJ/kgK)
        self.cp_water = 4.18

        # Modelo Detallado del Receptor (Tubo de acero inoxidable evacuado)
        self.tube_length = 21.0 # m
        # Masa proporcional al largo respecto a la tubería total
        length_ratio = self.tube_length / config['piping']['length_m']
        self.tube_mass = self.pipe_metal_mass * length_ratio
        # Metal de cañería que acompaña al nodo de agua de recirculación (excluye el receptor)
        self.pipe_metal_node = max(0.0, self.pipe_metal_mass - self.tube_mass)
        
        # Área interna para convección Tubo -> Fluido (asumimos D_int ~ 40mm)
        self.tube_area_inner = np.pi * 0.040 * self.tube_length
        self.tube_h_conv = 1500.0 # W/m2K (Convección forzada agua)

    def get_water_properties(self, P_bar, T_C=None, x=None):
        """Obtiene propiedades del agua/vapor usando IAPWS-IF97."""
        P_mpa = P_bar / 10.0
        if x is not None:
            state = IAPWS97(P=P_mpa, x=x)
        elif T_C is not None:
            state = IAPWS97(P=P_mpa, T=T_C + 273.15)
        else:
            # Saturado líquido por defecto si solo hay P
            state = IAPWS97(P=P_mpa, x=0)
        
        return {
            "h": state.h,     # kJ/kg
            "rho": state.rho, # kg/m3
            "u": state.u,     # kJ/kg
            "T": state.T - 273.15
        }

    def get_iam(self, theta_l, theta_t):
        """Calcula el factor IAM combinado por interpolación."""
        iam_config = self.config['collector'].get('iam', {})
        if not iam_config:
            return 1.0
            
        angles = iam_config['angles']
        iam_l = np.interp(abs(theta_l), angles, iam_config['longitudinal'])
        iam_t = np.interp(abs(theta_t), angles, iam_config['transversal'])
        return iam_l * iam_t

    def simulate_step(self, current_state, inputs, dt_min):
        """
        Ejecuta un paso de simulación avanzado con balances de masa y energía en el drum.
        inputs ahora debe incluir 'timestamp' para el cálculo solar.
        """
        dt = dt_min * 60.0
        P_amb = 1.013 # bar
        
        # --- 1. CAMPO SOLAR (Cálculo de ángulos e IAM) ---
        timestamp = inputs.get('timestamp')
        iam_factor = 1.0
        dni_effective = inputs.get('dni', 0.0)
        
        if PVLIB_AVAILABLE and timestamp is not None:
            try:
                # Configuración de ubicación
                loc = self.config['location']
                ts = pd.Timestamp(timestamp)
                
                # Asegurar que el timestamp tenga la zona horaria de la planta para pvlib
                if ts.tz is None:
                    ts = ts.tz_localize(loc.get('timezone', 'UTC'))
                
                sol_pos = get_solarposition(
                    ts, 
                    loc['latitude'], 
                    loc['longitude'], 
                    loc['altitude']
                )
                
                zen_deg = sol_pos['zenith'].values[0]
                zenith = np.radians(zen_deg)
                azimuth = np.radians(sol_pos['azimuth'].values[0])
                
                # --- Estimación de DNI a partir de GHI (Modelo DIRINT, Perez 1992) ---
                ghi = inputs.get('dni', 0.0)
                dni_effective = estimate_dni_perez(ghi, zen_deg, ts)

                # --- Ángulos Fresnel (E-W) ---
                theta_l = np.degrees(np.arcsin(np.sin(zenith) * np.sin(azimuth)))
                theta_t = np.degrees(np.arctan2(np.sin(zenith) * np.cos(azimuth), np.cos(zenith)))
                
                iam_factor = self.get_iam(theta_l, theta_t)
            except Exception:
                iam_factor = 1.0
                dni_effective = inputs.get('dni', 0.0)

        # --- 2. MODELO DE TUBO RECEPTOR (INERCIA) ---
        # T_tube es la temperatura del metal del receptor
        t_tube = current_state.get('T_tube', current_state['T_sf_in'])
        # T_pipe: temperatura del agua del lazo de recirculación (cañería + receptor).
        # Es un nodo propio: de noche (sin flujo) se enfría hacia el ambiente; al
        # arrancar la bomba se mezcla con el agua caliente del drum.
        t_pipe = current_state.get('T_pipe', current_state.get('T_drum', current_state['T_sf_in']))
        t_drum_prev = current_state.get('T_drum', current_state['T_sf_in'])
        
        # Energía absorbida: η₀ × optical_loss_factor (suciedad + tracking error) × IAM
        optical_loss = self.config['collector'].get('optical_loss_factor', 1.0)
        q_abs = dni_effective * self.area * inputs['eta_opt'] * optical_loss * iam_factor / 1000.0 # kW
        warmup_config = self.config['collector'].get('warmup', {})
        warmup_factor = 1.0
        if warmup_config.get('enabled', False):
            warmup_start = warmup_config.get('start_temp_C', 40.0)
            warmup_full = warmup_config.get('full_temp_C', 95.0)
            min_factor = warmup_config.get('min_factor', 0.25)
            if warmup_full > warmup_start:
                warmup_factor = min_factor + (1.0 - min_factor) * (
                    (t_tube - warmup_start) / (warmup_full - warmup_start)
                )
                warmup_factor = float(np.clip(warmup_factor, min_factor, 1.0))
        q_abs_effective = q_abs * warmup_factor
        
        # Pérdidas del tubo al ambiente (Radiación en vacío)
        # Usamos los coeficientes de pérdida sobre la T del tubo
        dT_tube_amb = t_tube - inputs['t_amb']
        a1, a2 = self.config['collector']['thermal_loss_coefficients']
        q_loss = self.area * (a1 * dT_tube_amb + a2 * (dT_tube_amb**2)) / 1000.0 # kW
        
        flow_active = inputs['m_rec'] > 0
        battery_config = self.config['collector'].get('thermal_battery', {})
        battery_enabled = battery_config.get('enabled', False)
        t_loop = current_state.get('T_loop', t_tube)
        battery_capacity = battery_config.get('capacity_kj_k', 0.0)
        battery_charge_fraction = battery_config.get('charge_fraction', 0.0) if battery_enabled else 0.0
        battery_charge_fraction = float(np.clip(battery_charge_fraction, 0.0, 0.8))
        q_battery_charge = q_abs_effective * battery_charge_fraction
        q_abs_tube = q_abs_effective - q_battery_charge
        
        # Transferencia Tubo -> Fluido (Agua del lazo de recirculación, a T_pipe)
        t_fluid_avg = t_pipe
        # Q_fluid = h * A * (T_tube - T_fluid)
        q_fluid = self.tube_h_conv * self.tube_area_inner * (t_tube - t_fluid_avg) / 1000.0 # kW
        
        # Balance en el tubo (Integración Implícita para evitar inestabilidad numérica)
        # M*Cp*(T_new - T_old)/dt = Q_abs - Q_loss - h*A*(T_new - T_fluid)
        # T_new = (M*Cp*T_old + dt*(Q_abs - Q_loss + h*A*T_fluid)) / (M*Cp + dt*h*A)
        m_cp_tube = self.tube_mass * self.pipe_cp
        h_A_tube_kW = self.tube_h_conv * self.tube_area_inner / 1000.0 if flow_active else 0.0 # kW/K
        
        t_tube_new = (m_cp_tube * t_tube + dt * (q_abs_tube - q_loss + h_A_tube_kW * t_fluid_avg)) / (m_cp_tube + dt * h_A_tube_kW)
        
        # Recalculamos Q_fluid real transferido en este paso con la nueva temperatura
        q_fluid = h_A_tube_kW * (t_tube_new - t_fluid_avg)
        
        q_battery_fluid = 0.0
        q_battery_loss = 0.0
        t_loop_new = t_loop
        if battery_enabled and battery_capacity > 0:
            ua_fluid_kw_k = battery_config.get('ua_fluid_w_k', 0.0) / 1000.0
            ua_loss_kw_k = battery_config.get('ua_loss_w_k', 0.0) / 1000.0
            ua_fluid_active = ua_fluid_kw_k if flow_active else 0.0
            denominator = battery_capacity + dt * (ua_fluid_active + ua_loss_kw_k)
            numerator = (
                battery_capacity * t_loop
                + dt * (
                    q_battery_charge
                    + ua_fluid_active * t_fluid_avg
                    + ua_loss_kw_k * inputs['t_amb']
                )
            )
            t_loop_new = numerator / denominator
            q_battery_fluid = ua_fluid_active * (t_loop_new - t_fluid_avg)
            q_battery_loss = ua_loss_kw_k * (t_loop_new - inputs['t_amb'])
        
        # Ganancia neta hacia el agua del lazo (puede ser negativa si el tubo está más frío)
        q_net_sf = (q_fluid + q_battery_fluid) if flow_active else 0.0

        # --- 2b. NODO DE AGUA DE CAÑERÍA / LAZO DE RECIRCULACIÓN (T_pipe) ---
        # Balance: C_pipe*dT_pipe/dt = Q_net_sf + m_rec*cp*(T_drum - T_pipe) - UA_pipe*(T_pipe - T_amb)
        #   - Sin flujo: el agua de cañería se enfría hacia el ambiente (decoupled del drum).
        #   - Con flujo: intercambia masa con el drum -> la mezcla fría/caliente del arranque.
        cp_w = self.cp_water
        try:
            rho_pipe = self.get_water_properties(current_state['P_drum'], T_C=t_pipe)['rho']
        except Exception:
            rho_pipe = 1000.0
        m_pipe_w = self.pipe_vol * rho_pipe
        c_pipe = m_pipe_w * cp_w + self.pipe_metal_node * self.pipe_cp        # kJ/K
        ua_pipe_kw = self.pipe_h_loss * self.pipe_area / 1000.0               # kW/K
        m_rec_coupling = inputs['m_rec'] * cp_w if flow_active else 0.0       # kW/K (advección lazo<->drum)

        denom_pipe = c_pipe / dt + m_rec_coupling + ua_pipe_kw
        numer_pipe = (c_pipe / dt) * t_pipe + q_net_sf \
            + m_rec_coupling * t_drum_prev + ua_pipe_kw * inputs['t_amb']
        t_pipe_new = numer_pipe / denom_pipe
        if not battery_enabled:
            t_loop_new = t_pipe_new  # when battery disabled, loop tracks pipe temperature

        # Calor neto que la recirculación entrega al drum (agua retorna a T_pipe, sale a T_drum)
        q_recirc_to_drum = m_rec_coupling * (t_pipe_new - t_drum_prev)        # kW

        # Salida del campo solar = temperatura del lazo (para reporte y comparación con T8)
        t_out_sf = t_pipe_new

        # Modelo de flujo por orificio (Válvula)
        kv_valve = 0.012 
        m_vapor = kv_valve * (inputs['valve_open'] / 100.0) * np.sqrt(max(0, current_state['P_drum'] - P_amb))
        
        # --- 3. BALANCE EN EL STEAM DRUM (Masa y Energía Interna) ---
        # Entalpías de entrada/salida
        # h_feed @ T_feed, P_drum
        h_feed = self.get_water_properties(current_state['P_drum'], T_C=inputs.get('T_feed', 20))['h']
        # h_vap (vapor saturado)
        h_vap = self.get_water_properties(current_state['P_drum'], x=1)['h']
        
        # Balances diferenciales (Euler simple)
        dm_dt = inputs['m_feed'] - m_vapor

        # Usar T_drum del paso anterior (resuelto por equilibrio termodinámico)
        # Correcto tanto en subenfriamiento como en condiciones bifásicas
        t_avg_drum = t_drum_prev

        # Pérdidas del drum al ambiente (la cañería pierde calor en su propio nodo T_pipe)
        q_loss_drum = self.drum_h_loss * self.drum_area * (t_avg_drum - inputs['t_amb']) / 1000.0

        # dU/dt = m_feed*h_feed + Q_recirc(lazo->drum) - m_vapor*h_vap - Q_loss_drum
        # El aporte solar entra al drum a través del retorno caliente de la recirculación.
        du_dt = (inputs['m_feed'] * h_feed) + q_recirc_to_drum - (m_vapor * h_vap) - q_loss_drum
        
        # Actualización de estados acumulados
        M_new = current_state['M_total'] + dm_dt * dt
        U_new = current_state['U_total'] + du_dt * dt
        
        # --- 4. CÁLCULO DE NUEVA PRESIÓN Y TEMPERATURA (Equilibrio Termodinámico) ---
        # Resolvemos T y P que satisfacen el volumen del drum y la energía total
        P_new, T_new, quality = self._solve_equilibrium(M_new, U_new, current_state.get('P_drum'))
        
        # Debug mass balance
        if inputs.get('debug'):
             print(f"DEBUG: Q_abs={q_abs:.1f}kW, Q_loss={q_loss:.1f}kW, m_v={m_vapor:.4f}, P={P_new:.2f}, x={quality:.4f}")

        # --- 5. CÁLCULO DE NIVEL ---
        props_f = self.get_water_properties(P_new, x=0)
        v_f = 1.0 / props_f['rho']
        
        M_liquid = M_new * (1 - quality)

        # M_total ahora representa SOLO el inventario del drum (la cañería es su propio
        # nodo), por lo que el nivel usa directamente el líquido del drum.
        V_liquid_drum = max(0, M_liquid * v_f)

        area_drum = np.pi * (self.drum_d/2)**2
        level_new = (V_liquid_drum / area_drum) * 1000.0 # mm
        
        return {
            "P_drum_sim": P_new,
            "T_drum_sim": T_new,
            "T_pipe_sim": t_pipe_new,
            "T_sf_out_sim": t_out_sf,
            "T_tube_sim": t_tube_new,
            "T_loop_sim": t_loop_new,
            "Q_recirc": q_recirc_to_drum,
            "level_sim": level_new,
            "m_vapor_sim": m_vapor,
            "M_total_new": M_new,
            "U_total_new": U_new,
            "Q_abs": q_abs,
            "Q_abs_effective": q_abs_effective,
            "Q_abs_tube": q_abs_tube,
            "Q_battery_charge": q_battery_charge,
            "Q_battery_fluid": q_battery_fluid,
            "Q_battery_loss": q_battery_loss,
            "warmup_factor": warmup_factor,
            "Q_loss": q_loss,
            "Q_net": q_net_sf,
            "dni_est": dni_effective,
            "eta_total": inputs['eta_opt'] * iam_factor
        }

    def _solve_equilibrium(self, M_total, U_total, P_hint_bar=None, max_iter=10):
        """
        Encuentra P y T de equilibrio para una masa M y energía U en un volumen fijo.
        Considera la inercia térmica del metal del drum.

        M_total / U_total representan SOLO el inventario del drum; la cañería se
        modela como un nodo aparte (T_pipe) en simulate_step.
        """
        V_total = self.drum_vol
        # Inercia del metal del drum (el receptor y la cañería tienen nodos propios)
        residual_metal_mass = self.drum_metal_mass
        
        # Subcooled liquid: do not force cold startup states onto saturation.
        P_liq_bar = max(1.013, P_hint_bar or 1.013)
        
        try:
            state_sat_liq = self.get_water_properties(P_liq_bar, x=0)
            U_sat_liq = M_total * state_sat_liq['u'] + residual_metal_mass * self.pipe_cp * state_sat_liq['T']
            if U_total < U_sat_liq:
                T_low = 0.01
                T_high = max(T_low, state_sat_liq['T'] - 0.01)
                
                for _ in range(30):
                    T_mid = 0.5 * (T_low + T_high)
                    state_liq = self.get_water_properties(P_liq_bar, T_C=T_mid)
                    U_mid = M_total * state_liq['u'] + residual_metal_mass * self.pipe_cp * T_mid
                    
                    if U_mid < U_total:
                        T_low = T_mid
                    else:
                        T_high = T_mid
                
                return P_liq_bar, 0.5 * (T_low + T_high), 0.0
        except Exception:
            pass
        
        T_guess = 100.0 + (U_total / (M_total * 4.18 + residual_metal_mass * self.pipe_cp)) * 0.1 
        if U_total > 5000: # Si ya hay energía considerable
            T_guess = 130.0
            
        # Solver simple (Secante o aproximación sucesiva)
        # Rango físico para T de saturación del agua (IAPWS-IF97): punto triple .. crítico.
        T_SAT_MIN, T_SAT_MAX = 0.01, 373.0
        for _ in range(max_iter):
            # Mantener T_guess dentro del dominio válido para evitar P_sat fuera de rango
            T_guess = min(T_SAT_MAX, max(T_SAT_MIN, T_guess))
            # 1. P_sat a T_guess
            try:
                from iapws.iapws97 import _PSat_T
                p_mpa = _PSat_T(T_guess + 273.15)
                P_bar = p_mpa * 10.0
            except:
                P_bar = max(1.013, (T_guess / 100.0)**4)
            
            # 2. Propiedades en saturación
            state_f = self.get_water_properties(P_bar, x=0)
            state_g = self.get_water_properties(P_bar, x=1)
            
            v_f = 1.0 / state_f['rho']
            v_g = 1.0 / state_g['rho']
            u_f = state_f['u']
            u_g = state_g['u']
            
            # 3. Calidad de vapor por volumen
            v_avg = V_total / M_total
            quality = (v_avg - v_f) / (v_g - v_f)
            quality = max(0, min(1, quality))
            
            # 4. Energía interna del agua
            u_water = u_f + quality * (u_g - u_f)
            
            # 5. Energía total calculada con este T_guess
            U_calc = M_total * u_water + residual_metal_mass * self.pipe_cp * T_guess
            
            # 6. Ajuste de T_guess (derivada numérica dU/dT para convergencia robusta en bifásico)
            error_U = U_total - U_calc
            dT_num = 0.1  # K perturbación para derivada numérica
            try:
                p_plus_mpa = _PSat_T(T_guess + dT_num + 273.15)
                P_plus = p_plus_mpa * 10.0
            except:
                P_plus = max(1.013, ((T_guess + dT_num) / 100.0)**4)
            sf_p = self.get_water_properties(P_plus, x=0)
            sg_p = self.get_water_properties(P_plus, x=1)
            v_f_p, v_g_p = 1.0 / sf_p['rho'], 1.0 / sg_p['rho']
            q_p = max(0, min(1, (v_avg - v_f_p) / (v_g_p - v_f_p)))
            u_w_p = sf_p['u'] + q_p * (sg_p['u'] - sf_p['u'])
            U_calc_plus = M_total * u_w_p + residual_metal_mass * self.pipe_cp * (T_guess + dT_num)
            
            dU_dT = (U_calc_plus - U_calc) / dT_num
            if abs(dU_dT) > 0.01:
                T_guess += error_U / dU_dT
            else:
                # Fallback para condiciones extremas
                T_guess += error_U / (M_total * 4.18 + residual_metal_mass * self.pipe_cp)
            
            if abs(error_U) < 1.0: # Convergencia en 1 kJ
                break
                
        return max(1.013, P_bar), T_guess, quality
