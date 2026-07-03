import pandas as pd
import os
import glob

DATA_DIR = "data"

REQUESTED_COLS = [
    'timestamp',
    'RAD', 'T_AMB',
    'T5', 'T8', 'P8',
    'T3', 'T2',
    'T4', 'P4',
    'P5', 'CAUDAL', 'EV FB', 'CAUDAL VAP', 'Nivel'
]

CONTINUOUS_SENSOR_COLS = ['T_AMB', 'T5', 'T8', 'T3', 'T2', 'T4', 'P8', 'P4', 'P5']


def smooth_sensor_outliers(df, columns, window=7, n_sigmas=4.0):
    """Replace isolated sensor spikes using a rolling Hampel filter."""
    cleaned = df.copy()
    for col in columns:
        if col not in cleaned.columns:
            continue
        series = pd.to_numeric(cleaned[col], errors='coerce')
        rolling_median = series.rolling(window=window, center=True, min_periods=3).median()
        abs_deviation = (series - rolling_median).abs()
        rolling_mad = abs_deviation.rolling(window=window, center=True, min_periods=3).median()
        threshold = n_sigmas * 1.4826 * rolling_mad
        outliers = (abs_deviation > threshold) & threshold.notna() & (threshold > 0)
        outlier_count = int(outliers.sum())
        if outlier_count:
            print(f"    Smoothed {outlier_count} outliers in {col}.")
            cleaned.loc[outliers, col] = rolling_median.loc[outliers]
    return cleaned


def process_file(input_path, output_csv, output_parquet):
    print(f"  Procesando {input_path}...")
    df = pd.read_csv(input_path)

    df = df.rename(columns={'_time': 'timestamp'})
    cols_to_drop = ['_field', 'key', 'objectType']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Consolidar filas dispersas por timestamp
    df_proc = df.groupby('timestamp').first().reset_index()

    # Asegurar columnas requeridas
    for col in REQUESTED_COLS:
        if col not in df_proc.columns:
            print(f"    [AVISO] Columna '{col}' no encontrada. Se agrega como NaN.")
            df_proc[col] = pd.NA

    df_proc = df_proc[REQUESTED_COLS]

    # Zona horaria UTC → America/Santiago
    df_proc['timestamp'] = pd.to_datetime(df_proc['timestamp'])
    if df_proc['timestamp'].dt.tz is None:
        df_proc['timestamp'] = df_proc['timestamp'].dt.tz_localize('UTC')
    df_proc['timestamp'] = df_proc['timestamp'].dt.tz_convert('America/Santiago')
    df_proc = df_proc.sort_values('timestamp')

    # ── resampleo a 1 min si el muestreo original es > 90 s ──────────────────
    df_proc = df_proc.set_index('timestamp')
    if len(df_proc) > 1:
        dt_median = df_proc.index.to_series().diff().median()
        if dt_median > pd.Timedelta('90s'):
            print(f"    Muestreo detectado: {dt_median}. Interpolando a 1 min...")
            df_proc = df_proc.resample('1min').asfreq()
            # Columnas continuas: interpolación lineal por tiempo
            interp_cols = [c for c in df_proc.columns
                           if c in REQUESTED_COLS and c != 'timestamp']
            df_proc[interp_cols] = df_proc[interp_cols].interpolate(method='time')
            # RAD no puede ser negativo
            if 'RAD' in df_proc.columns:
                df_proc['RAD'] = df_proc['RAD'].clip(lower=0.0)
            print(f"    Filas tras resampleo: {len(df_proc)}")
    df_proc = df_proc.reset_index()

    df_proc = smooth_sensor_outliers(df_proc, CONTINUOUS_SENSOR_COLS)

    df_proc.to_csv(output_csv, index=False)
    df_proc.to_parquet(output_parquet, index=False)
    print(f"    Guardado: {output_csv}  ({len(df_proc)} filas, "
          f"{df_proc['timestamp'].min().strftime('%H:%M')}–"
          f"{df_proc['timestamp'].max().strftime('%H:%M')})")


def main():
    raw_files = sorted(glob.glob(os.path.join(DATA_DIR, "raw_*.csv")))

    if not raw_files:
        print(f"No se encontraron archivos raw_*.csv en '{DATA_DIR}/'.")
        print("Ejecuta primero scripts/download_data.py")
        return

    print(f"Archivos a procesar: {len(raw_files)}")
    for raw_path in raw_files:
        # raw_2026-04-22.csv  →  processed_2026-04-22.csv
        basename = os.path.basename(raw_path)          # raw_2026-04-22.csv
        date_str = basename.replace("raw_", "").replace(".csv", "")  # 2026-04-22
        out_csv     = os.path.join(DATA_DIR, f"processed_{date_str}.csv")
        out_parquet = os.path.join(DATA_DIR, f"processed_{date_str}.parquet")
        process_file(raw_path, out_csv, out_parquet)

    print("Procesamiento completo.")


if __name__ == "__main__":
    base_path = r"c:\Users\Pablo Castillo\OneDrive - fraunhofer.cl\Documentos\01. RESEARCH\10. DSG_Caren\digital_shadow"
    os.chdir(base_path)
    main()
