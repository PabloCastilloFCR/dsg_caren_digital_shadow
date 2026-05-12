import requests
import json
from datetime import datetime

url = "http://localhost:8000/predict"

# Datos de prueba simulando el PLC
payload = {
    "timestamp": datetime.now().isoformat(),
    "ghi": 800.0,
    "t_amb": 25.0,
    "p_drum": 12.0,
    "t_drum_bottom": 185.0,
    "m_feed": 0.5,
    "m_rec": 2.0,
    "m_vapor": 0.4,
    "valve_pos": 45.0,
    "t_sf_in": 180.0,
    "t_sf_out": 210.0
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e)
