"""Lógica de control de presión del drum (válvula de alivio de vapor).

Replica el controlador real del fabricante: un proporcional puro con banda
fija. La válvula permanece cerrada hasta que la presión alcanza el setpoint y
abre linealmente hasta 100% cuando la presión supera el setpoint en `offset`.

    %_apertura = clamp((P_actual - P_setpoint) / offset, 0, 1) * 100

Ejemplo (P_setpoint = 2.0 bar, offset = 0.5 bar):
    P_actual = 2.0 bar -> 0%   (inicia apertura)
    P_actual = 2.1 bar -> 20%  (0.1 / 0.5)
    P_actual = 2.5 bar -> 100% (0.5 / 0.5, saturado)
"""

# Banda proporcional por defecto del fabricante (bar).
DEFAULT_OFFSET_BAR = 0.5


def compute_valve_opening(p_actual_bar, p_setpoint_bar, offset_bar=DEFAULT_OFFSET_BAR):
    """Devuelve el porcentaje de apertura de la válvula (0-100).

    Parameters
    ----------
    p_actual_bar : float
        Presión actual del drum.
    p_setpoint_bar : float
        Presión objetivo a la que la válvula empieza a abrir.
    offset_bar : float
        Banda proporcional: presión por encima del setpoint para la cual la
        válvula alcanza el 100% de apertura. Debe ser > 0.

    Notes
    -----
    `p_actual_bar` y `p_setpoint_bar` deben usar la misma referencia
    (ambas absolutas o ambas manométricas); la diferencia es lo que importa.
    """
    if offset_bar <= 0:
        raise ValueError("offset_bar debe ser > 0")

    fraction = (p_actual_bar - p_setpoint_bar) / offset_bar
    return max(0.0, min(1.0, fraction)) * 100.0
