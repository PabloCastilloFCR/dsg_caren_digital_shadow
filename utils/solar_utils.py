import numpy as np
import pandas as pd
from pvlib import solarposition, irradiance, tracking

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
        """Estima DNI usando el modelo Erbs."""
        times = pd.to_datetime([timestamp]).tz_localize(self.tz)
        solpos = solarposition.get_solarposition(times, self.lat, self.lon, self.alt)
        
        # Usar modelo de Erbs para separar DNI/DHI de GHI
        dni_est = irradiance.erbs(ghi, solpos.zenith, times.dayofyear)['dni']
        return dni_est.values[0]

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
