import pandas as pd
import os

def process_plant_data(input_file, output_csv, output_parquet):
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Rename time column
    df = df.rename(columns={'_time': 'timestamp'})
    
    # Drop metadata columns from InfluxDB export
    cols_to_drop = ['_field', 'key', 'objectType']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Consolidate sparse data by timestamp
    # Group by timestamp and take the first non-null value for each column
    print("Consolidating sparse data...")
    df_processed = df.groupby('timestamp').first().reset_index()
    
    # Ensure all requested columns exist (fill with NaN if missing)
    requested_cols = ['timestamp', 'RAD', 'T_AMB', 'T5', 'T8', 'T3', 'T4', 'P5', 'CAUDAL', 'EV FB', 'CAUDAL VAP', 'Nivel']
    for col in requested_cols:
        if col not in df_processed.columns:
            print(f"Warning: Column {col} not found in data. Adding as NaN.")
            df_processed[col] = pd.NA
            
    # Reorder columns
    df_processed = df_processed[requested_cols]
    
    # Sort by timestamp and handle timezone (UTC to Chile)
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    if df_processed['timestamp'].dt.tz is None:
        df_processed['timestamp'] = df_processed['timestamp'].dt.tz_localize('UTC')
    df_processed['timestamp'] = df_processed['timestamp'].dt.tz_convert('America/Santiago')
    
    df_processed = df_processed.sort_values('timestamp')
    
    # Save to CSV
    print(f"Saving to {output_csv}...")
    df_processed.to_csv(output_csv, index=False)
    
    # Save to Parquet
    print(f"Saving to {output_parquet}...")
    df_processed.to_parquet(output_parquet, index=False)
    
    print("Processing complete!")
    print(f"Shape of processed data: {df_processed.shape}")
    print("\nFirst 5 rows:")
    print(df_processed.head())

if __name__ == "__main__":
    base_path = r"c:\Users\Pablo Castillo\OneDrive - fraunhofer.cl\Documentos\01. RESEARCH\10. DSG_Caren\digital_shadow"
    input_csv = os.path.join(base_path, "data", "plant_data_april.csv")
    output_csv = os.path.join(base_path, "data", "plant_data_processed.csv")
    output_parquet = os.path.join(base_path, "data", "plant_data_processed.parquet")
    
    process_plant_data(input_csv, output_csv, output_parquet)
