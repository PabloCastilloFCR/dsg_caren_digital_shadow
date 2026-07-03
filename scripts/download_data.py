from influxdb_client import InfluxDBClient
import pandas as pd
import os

# Configuración de conexión
INFLUX_URL = "http://192.168.2.195:8086"
INFLUX_TOKEN = "f5R5OAo8UboGLOtkgbSTLdGXiyoJOxX0eObDOBMVudtVjo4svHh3xU-oLnBV_b1-7jgUzyyYxgFnghh4Ib_ItQ=="
INFLUX_ORG = "fraunhofer"
BUCKET = "caren"
MEASUREMENT = "plc_analog"
WINDOW = "1m"

# Días a descargar — agregar o quitar fechas aquí
DATES = [
    "2026-04-22",
    "2026-05-14",
    "2026-06-14",
    "2026-04-13",
]

SENSORS = [
    "RAD", "T_AMB",
    "T5", "P5", "T8", "P8",
    "T4", "P4", "T2", "T3", "EV FB",
    "CAUDAL", "CAUDAL VAP", "Nivel"
]

OUTPUT_DIR = "data"


def build_flux_query(sensors, start, stop, window):
    filters = " or ".join([f'r["objectName"] == "{name}"' for name in sensors])
    return f'''
    from(bucket: "{BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
      |> filter(fn: (r) => {filters})
      |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
      |> pivot(rowKey: ["_time"], columnKey: ["objectName"], valueColumn: "_value")
    '''


def download_day(query_api, date_str):
    start = f"{date_str}T00:00:00Z"
    stop  = f"{date_str}T23:59:59Z"
    print(f"  Descargando {date_str}...")

    query = build_flux_query(SENSORS, start, stop, WINDOW)
    df = query_api.query_data_frame(query)

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)

    if df.empty:
        print(f"  [AVISO] Sin datos para {date_str}.")
        return

    cols_to_drop = [c for c in ["result", "table", "_start", "_stop", "_measurement"] if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    output_path = os.path.join(OUTPUT_DIR, f"raw_{date_str}.csv")
    df.to_csv(output_path, index=False)
    print(f"  Guardado: {output_path}  ({len(df)} filas)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        query_api = client.query_api()
        for date_str in DATES:
            download_day(query_api, date_str)
    print("Descarga completa.")


if __name__ == "__main__":
    main()
