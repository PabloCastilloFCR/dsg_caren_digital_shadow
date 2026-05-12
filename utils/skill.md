# Skill: Descarga de Datos desde InfluxDB con Python

Esta guía proporciona los pasos necesarios para descargar y procesar datos almacenados en la base de datos local de **InfluxDB** utilizando el script `influxdb_plot.py`.

## 1. Configuración del Entorno

Antes de comenzar, asegúrate de tener instaladas las dependencias necesarias.

```bash
pip install influxdb-client pandas matplotlib
```

O usa el archivo de requerimientos:
```bash
pip install -r requirements.txt
```

## 2. Parámetros de Descarga

En el archivo `influxdb_plot.py`, puedes personalizar qué datos descargar modificando la sección de **CONFIGURACIÓN**:

*   **Rango de Tiempo (`START`, `STOP`)**: 
    *   `-1h`: Última hora.
    *   `-24h`: Último día.
    *   `-7d`: Última semana.
*   **Resolución (`WINDOW`)**: Define cada cuánto tiempo se promedian los datos (ej: `1m`, `5m`, `1h`). Esto evita saturar la memoria con demasiados puntos.
*   **Sensores (`TEMP_SENSORS`)**: Lista de nombres de objetos (Tags) que deseas extraer (ej: `["T1", "T2"]`).

## 3. Proceso de Descarga (Lógica del Script)

La descarga se realiza en tres etapas técnicas detalladas en el script:

1.  **Conexión**: Se establece el túnel con el servidor `http://192.168.2.195:8086` usando un Token de seguridad.
2.  **Consulta (Flux)**: Se envía una instrucción al servidor para filtrar los datos.
    ```flux
    from(bucket: "caren")
      |> range(start: -8h)
      |> filter(fn: (r) => r["_measurement"] == "plc_analog")
      |> pivot(rowKey: ["_time"], columnKey: ["objectName"], valueColumn: "_value")
    ```
3.  **Conversión a DataFrame**: Los datos llegan en formato crudo y se transforman en una tabla de **Pandas** para facilitar su manipulación.

## 4. Cómo Exportar los Datos a Excel/CSV

Si además de visualizar los datos quieres guardarlos en tu computadora, puedes añadir estas líneas al final del script `influxdb_plot.py`:

```python
# Guardar datos de temperatura en un archivo CSV
temp_df.to_csv("datos_temperatura.csv")

# Guardar en formato Excel (requiere pip install openpyxl)
# temp_df.to_excel("datos_monitoreo.xlsx")
```

## 5. Solución de Problemas Comunes

*   **Error de Conexión**: Asegúrate de estar conectado a la red local (VPN o WiFi de la planta) ya que la IP `192.168.2.195` es privada.
*   **Consulta sin Datos**: Verifica que el nombre del sensor en `TEMP_SENSORS` coincida exactamente con el registrado en la base de datos (sensible a mayúsculas).
*   **Token Expirado**: Si el script falla con un error 401 (Unauthorized), solicita un nuevo Token al administrador del sistema.

---
> [!TIP]
> Para consultas de larga duración (ej: un mes de datos), aumenta el valor de `WINDOW` a `1h` para reducir el tiempo de descarga y el uso de RAM.
