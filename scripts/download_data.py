from influxdb_client import InfluxDBClient
import pandas as pd
import os

# Configuración
INFLUX_URL = "http://192.168.2.195:8086"
INFLUX_TOKEN = "f5R5OAo8UboGLOtkgbSTLdGXiyoJOxX0eObDOBMVudtVjo4svHh3xU-oLnBV_b1-7jgUzyyYxgFnghh4Ib_ItQ=="
INFLUX_ORG = "fraunhofer"
BUCKET = "caren"
MEASUREMENT = "plc_analog"

# Rango de tiempo solicitado: 21 al 24 de abril de 2026
START = "2026-04-22T00:00:00Z"
STOP = "2026-04-23T00:00:00Z"
WINDOW = "1m" # 1 minuto de resolución para el Digital Twin

# Sensores
SENSORS = [
    "RAD", "T_AMB", 
    "T5", "P5", "T8","T4", "T2", "T3", "EV FB",
    "CAUDAL", "CAUDAL VAP", "Nivel"
]

def build_flux_query(sensors, start, stop, window):
    filters = " or ".join([f'r["objectName"] == "{name}"' for name in sensors])
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
      |> filter(fn: (r) => {filters})
      |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
      |> pivot(rowKey: ["_time"], columnKey: ["objectName"], valueColumn: "_value")
    '''
    return query

def main():
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        query_api = client.query_api()
        print(f"Descargando datos desde {START} hasta {STOP}...")
        
        query = build_flux_query(SENSORS, START, STOP, WINDOW)
        df = query_api.query_data_frame(query)
        
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
            
        if df.empty:
            print("No se encontraron datos.")
            return

        # Limpieza
        cols_to_drop = [c for c in ["result", "table", "_start", "_stop", "_measurement"] if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        # Guardar
        output_path = "data/plant_data_april.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en {output_path}")

if __name__ == "__main__":
    main()
