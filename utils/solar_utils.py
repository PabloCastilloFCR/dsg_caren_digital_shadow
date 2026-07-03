import numpy as np
import pandas as pd
from pvlib import solarposition, irradiance, tracking


def estimate_dni_perez(ghi, solar_zenith_deg, timestamp, pressure_pa=101325):
    """
    Estima DNI desde GHI usando el modelo DIRINT (Perez et al. 1992).

    A diferencia de Erbs (que usa solo el índice de claridad instantáneo kt),
    DIRINT considera la variación temporal de kt entre pasos consecutivos,
    lo que mejora la estimación en cielos despejados de alta irradiancia directa
    como los de la zona central de Chile.

    Args:
        ghi             : Irradiancia horizontal global (W/m²)
        solar_zenith_deg: Ángulo cenital solar (grados)
        timestamp       : pd.Timestamp con zona horaria
        pressure_pa     : Presión atmosférica en Pa (default 101325 = nivel del mar)

    Returns:
        dni (float): Irradiancia normal directa estimada (W/m²), >= 0
    """
    if ghi <= 0 or solar_zenith_deg >= 88:
        return 0.0
    try:
        idx = pd.DatetimeIndex([pd.Timestamp(timestamp)])
        ghi_s = pd.Series([float(ghi)], index=idx)
        zen_s = pd.Series([float(solar_zenith_deg)], index=idx)
        dni_s = irradiance.dirint(
            ghi_s, zen_s, idx,
            pressure=pressure_pa,
            use_delta_kt_prime=False,   # Paso único: sin serie temporal previa
        )
        return max(0.0, float(dni_s.iloc[0]))
    except Exception:
        return 0.0


class SolarUtility:
    def __init__(self, lat, lon, alt, tz):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.tz = tz

    def get_solar_state(self, timestamp):
        """Calcula la posición del sol y estima DNI a partir de GHI."""
        times = pd.to_datetime([timestamp]).tz_localize(self.tz)
        solpos = solarposition.get_solarposition(times, self.lat, self.lon, self.alt)
        
        zenith = solpos.zenith.values[0]
        azimuth = solpos.azimuth.values[0]
        
        return {
            "zenith": zenith,
            "azimuth": azimuth,
            "elevation": 90 - zenith
        }

    def estimate_dni(self, ghi, timestamp):
        """Estima DNI usando el modelo DIRINT (Perez et al. 1992)."""
        times = pd.to_datetime([timestamp])
        if times.tz is None:
            times = times.tz_localize(self.tz)
        solpos = solarposition.get_solarposition(times, self.lat, self.lon, self.alt)
        zen = float(solpos['zenith'].values[0])
        return estimate_dni_perez(ghi, zen, times[0])

    def get_fresnel_efficiency(self, timestamp, eta0, iam_l_list, iam_t_list):
        """
        Calcula la eficiencia óptica considerando el ángulo de incidencia.
        Asume seguimiento Norte-Sur (típico en Fresnel).
        """
        times = pd.to_datetime([timestamp]).tz_localize(self.tz)
        solpos = solarposition.get_solarposition(times, self.lat, self.lon, self.alt)
        
        # Seguimiento de un eje (Fresnel)
        # axis_tilt=0 (horizontal), axis_azimuth=0 (North-South)
        tru_track = tracking.singleaxis(solpos.zenith, solpos.azimuth, 
                                        axis_tilt=0, axis_azimuth=0)
        
        aoi = tru_track['aoi'].values[0]
        
        # Simplificación de IAM (interpolar o usar valor cercano por ahora)
        # En una versión avanzada usaríamos interpolación 2D.
        iam = np.cos(np.radians(aoi)) # Aproximación básica de Lambert si no hay datos
        
        return eta0 * iam
