import numpy as np
from iapws import IAPWS97

try:
    import pandas as pd
    from pvlib.solarposition import get_solarposition
    from pvlib.irradiance import erbs
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

        # Modelo Detallado del Receptor (Tubo de acero inoxidable evacuado)
        self.tube_length = 21.0 # m
        # Masa proporcional al largo respecto a la tubería total
        length_ratio = self.tube_length / config['piping']['length_m']
        self.tube_mass = self.pipe_metal_mass * length_ratio 
        
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
                
                # --- Estimación de DNI a partir de GHI (Modelo de Erbs) ---
                ghi = inputs.get('dni', 0.0)
                if ghi > 0 and zen_deg < 88:
                    doy = ts.dayofyear
                    erbs_data = erbs(ghi, zen_deg, doy)
                    dni_effective = erbs_data['dni']
                else:
                    dni_effective = 0.0

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
        
        # Energía absorbida por el tubo (95% absorción + IAM)
        q_abs = dni_effective * self.area * inputs['eta_opt'] * iam_factor / 1000.0 # kW
        
        # Pérdidas del tubo al ambiente (Radiación en vacío)
        # Usamos los coeficientes de pérdida sobre la T del tubo
        dT_tube_amb = t_tube - inputs['t_amb']
        a1, a2 = self.config['collector']['thermal_loss_coefficients']
        q_loss = self.area * (a1 * dT_tube_amb + a2 * (dT_tube_amb**2)) / 1000.0 # kW
        
        # Transferencia Tubo -> Fluido (Agua)
        t_fluid_avg = (current_state['T_sf_in'] + current_state['T_sf_out']) / 2
        # Q_fluid = h * A * (T_tube - T_fluid)
        q_fluid = self.tube_h_conv * self.tube_area_inner * (t_tube - t_fluid_avg) / 1000.0 # kW
        
        # Balance en el tubo (Integración Implícita para evitar inestabilidad numérica)
        # M*Cp*(T_new - T_old)/dt = Q_abs - Q_loss - h*A*(T_new - T_fluid)
        # T_new = (M*Cp*T_old + dt*(Q_abs - Q_loss + h*A*T_fluid)) / (M*Cp + dt*h*A)
        m_cp_tube = self.tube_mass * self.pipe_cp
        h_A_tube_kW = self.tube_h_conv * self.tube_area_inner / 1000.0 # kW/K
        
        t_tube_new = (m_cp_tube * t_tube + dt * (q_abs - q_loss + h_A_tube_kW * t_fluid_avg)) / (m_cp_tube + dt * h_A_tube_kW)
        
        # Recalculamos Q_fluid real transferido en este paso con la nueva temperatura
        q_fluid = h_A_tube_kW * (t_tube_new - t_fluid_avg)
        
        # Ganancia neta para el fluido (puede ser negativa si el tubo está más frío)
        q_net_sf = q_fluid
        
        # Enthalpy out of the solar field (h_sf)
        h_rec = self.get_water_properties(current_state['P_drum'], x=0)['h']
        if inputs['m_rec'] > 0:
            h_sf = h_rec + (q_net_sf / inputs['m_rec'])
        else:
            h_sf = h_rec # No flow
            
        # Estimación de T_out_sf usando IAPWS (maneja cambio de fase correctamente)
        if inputs['m_rec'] > 0:
            try:
                P_mpa = current_state['P_drum'] / 10.0
                state_out = IAPWS97(P=P_mpa, h=h_sf)
                t_out_sf = state_out.T - 273.15
            except:
                # Fallback si h_sf está fuera de rango IAPWS
                t_out_sf = current_state['T_sf_in'] + (q_net_sf / (inputs['m_rec'] * 4.18))
        else:
            t_out_sf = t_tube_new # Sin flujo, el fluido se estabiliza a la T del tubo
            
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
        
        # Estimación de temperatura actual para pérdidas
        # Inercia residual (excluyendo el receptor)
        residual_metal_mass = (self.pipe_metal_mass - self.tube_mass) + self.drum_metal_mass
        # Usar T_drum del paso anterior (resuelto por equilibrio termodinámico)
        # Correcto tanto en subenfriamiento como en condiciones bifásicas
        t_avg_drum = current_state.get('T_drum', current_state['T_sf_in'])
        
        # Pérdidas totales (Tuberías + Drum)
        q_loss_pipe = self.pipe_h_loss * self.pipe_area * (t_avg_drum - inputs['t_amb']) / 1000.0
        q_loss_drum = self.drum_h_loss * self.drum_area * (t_avg_drum - inputs['t_amb']) / 1000.0
        q_loss_total = q_loss_pipe + q_loss_drum
        
        # dU/dt = m_feed*h_feed + Q_net_sf - m_vapor*h_vap - Q_loss_total
        du_dt = (inputs['m_feed'] * h_feed) + q_net_sf - (m_vapor * h_vap) - q_loss_total
        
        # Actualización de estados acumulados
        M_new = current_state['M_total'] + dm_dt * dt
        U_new = current_state['U_total'] + du_dt * dt
        
        # --- 4. CÁLCULO DE NUEVA PRESIÓN Y TEMPERATURA (Equilibrio Termodinámico) ---
        # Resolvemos T y P que satisfacen el volumen del drum y la energía total
        P_new, T_new, quality = self._solve_equilibrium(M_new, U_new)
        
        # Debug mass balance
        if inputs.get('debug'):
             print(f"DEBUG: Q_abs={q_abs:.1f}kW, Q_loss={q_loss:.1f}kW, m_v={m_vapor:.4f}, P={P_new:.2f}, x={quality:.4f}")

        # --- 5. CÁLCULO DE NIVEL ---
        props_f = self.get_water_properties(P_new, x=0)
        v_f = 1.0 / props_f['rho']
        
        M_liquid = M_new * (1 - quality)
        
        # El nivel solo depende del líquido que está en el DRUM
        # Asumimos que la tubería siempre está llena de líquido (recirculación)
        V_liquid_total = M_liquid * v_f
        V_liquid_drum = max(0, V_liquid_total - self.pipe_vol)
        
        area_drum = np.pi * (self.drum_d/2)**2
        level_new = (V_liquid_drum / area_drum) * 1000.0 # mm
        
        return {
            "P_drum_sim": P_new,
            "T_drum_sim": T_new,
            "T_sf_out_sim": t_out_sf,
            "T_tube_sim": t_tube_new,
            "level_sim": level_new,
            "m_vapor_sim": m_vapor,
            "M_total_new": M_new,
            "U_total_new": U_new,
            "Q_abs": q_abs,
            "Q_loss": q_loss,
            "Q_net": q_net_sf,
            "dni_est": dni_effective,
            "eta_total": inputs['eta_opt'] * iam_factor
        }

    def _solve_equilibrium(self, M_total, U_total, max_iter=10):
        """
        Encuentra P y T de equilibrio para una masa M y energía U en un volumen fijo.
        Considera la inercia térmica del metal residual (Drum + conexiones).
        """
        V_total = self.drum_vol + self.pipe_vol
        # Inercia residual (excluyendo el receptor que se calcula dinámicamente)
        residual_metal_mass = (self.pipe_metal_mass - self.tube_mass) + self.drum_metal_mass
        
        # Estimación inicial de T
        T_guess = 100.0 + (U_total / (M_total * 4.18 + residual_metal_mass * self.pipe_cp)) * 0.1 
        if U_total > 5000: # Si ya hay energía considerable
            T_guess = 130.0
            
        # Solver simple (Secante o aproximación sucesiva)
        for _ in range(max_iter):
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
