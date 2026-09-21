# Digital Shadow — DSG Caren

Modelo de simulación térmica e hidráulica de una planta DSG (Direct Steam Generation) con campo solar Fresnel, steam drum, recirculación y control de presión/nivel.

El repositorio incluye datos de planta en CSV y Parquet, configuraciones del modelo y ejemplos de simulación sintética y validación con datos reales.

## Clonar y preparar el entorno

Requiere Python 3.9 o superior.

```powershell
git clone https://github.com/PabloCastilloFCR/digital_shadow.git
cd digital_shadow

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Linux o macOS, activa el entorno con:

```bash
source .venv/bin/activate
```

Ejecuta los comandos desde la carpeta raíz del repositorio, porque los scripts usan rutas relativas como `config/` y `data/`.

## Simulación sintética

Ejecuta una simulación de 120 pasos de un escenario con radiación solar constante, control proporcional de la válvula y control de nivel:

```powershell
python tests/synthetic_test.py
```

El ejemplo usa `config/plant_config.json`, muestra gráficos de presión, apertura de válvula, temperaturas, nivel y producción de vapor, y permanece abierto hasta cerrar la ventana del gráfico.

## Simulación con datos reales

Los datos procesados necesarios ya están incluidos en `data/`, por lo que no es necesario descargar nada para ejecutar este ejemplo:

```powershell
python tests/real_data_test.py
```

El script:

- carga `data/processed_2026-04-22.parquet` (o el CSV equivalente si no está disponible Parquet);
- ejecuta `model.plant_simulator.PlantSimulator` usando `config/plant_config_v2.json`;
- compara variables reales y simuladas;
- guarda el análisis en `tests/energy_gain_analysis_2026-04-22.csv`;
- guarda el gráfico en `tests/validation_2026-04-22.png`.

Para simular otra fecha, cambia `SIM_DATE` al inicio de `tests/real_data_test.py`. Las fechas disponibles corresponden a los archivos `processed_*.csv` y `processed_*.parquet` de `data/`.

## Pruebas y otros ejemplos

Prueba básica del balance de energía del modelo:

```powershell
python tests/test_thermal_model.py
```

Comparación de los factores IAM con el coseno teórico:

```powershell
python scripts/check_iam_cosine.py
```

Validación del modelo anterior con `data/plant_data_april.csv`:

```powershell
python scripts/validate_model.py
```

## Estructura del repositorio

```text
config/     Configuraciones de la planta y del colector
data/       Datos reales procesados y datos de ejemplo
model/      Modelos térmicos, simulador de planta y controlador
scripts/    Procesamiento, descarga y validación de datos
tests/      Simulaciones de ejemplo, pruebas y resultados
utils/      Utilidades solares
```

## Datos nuevos

La descarga desde InfluxDB está implementada en `scripts/download_data.py` y requiere acceso a la red interna de la planta y sus credenciales. Para trabajar únicamente con los datos ya versionados, no es necesario ejecutar ese script.

Los archivos Parquet están versionados intencionalmente para facilitar la transferencia completa del repositorio.
