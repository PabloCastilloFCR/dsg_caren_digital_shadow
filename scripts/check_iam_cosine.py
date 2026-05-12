import numpy as np
import matplotlib.pyplot as plt
import json
import os

def check_iam():
    # Cargar configuración
    config_path = 'config/plant_config.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} no encontrado.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    iam_config = config['collector'].get('iam', {})
    if not iam_config:
        print("Error: No se encontró configuración de IAM.")
        return

    angles = np.array(iam_config['angles'])
    iam_l = np.array(iam_config['longitudinal'])
    iam_t = np.array(iam_config['transversal'])
    cos_theta = np.cos(np.radians(angles))

    plt.figure(figsize=(12, 6))
    
    # Plot Longitudinal
    plt.subplot(1, 2, 1)
    plt.plot(angles, iam_l, 'bo-', label='IAM Longitudinal')
    plt.plot(angles, cos_theta, 'r--', label='Coseno Teórico')
    plt.fill_between(angles, iam_l, cos_theta, color='blue', alpha=0.1)
    plt.xlabel('Ángulo (°)')
    plt.ylabel('Factor')
    plt.title('IAM Longitudinal vs Coseno')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Transversal
    plt.subplot(1, 2, 2)
    plt.plot(angles, iam_t, 'go-', label='IAM Transversal (Tabla)')
    plt.plot(angles, cos_theta, 'r--', label='Coseno Teórico', alpha=0.5)
    
    # Nueva fórmula geométrica corregida: 1 - (h/L * tan(theta))
    h = 3.5
    L = 21
    with np.errstate(divide='ignore', invalid='ignore'):
        f_geom = 1 - (h * np.tan(np.radians(angles)) / L)
    
    # Ajustes de borde
    f_geom = np.maximum(0, f_geom)
    f_geom[angles >= 90] = 0
    
    plt.plot(angles, f_geom, 'm--', label=f'Fórmula Geométrica (h={h}, L={L})')
    plt.fill_between(angles, iam_t, f_geom, color='magenta', alpha=0.1)
    
    plt.xlabel('Ángulo (°)')
    plt.ylabel('Factor')
    plt.title('IAM Transversal vs Geometría')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'scripts/iam_vs_cosine.png'
    plt.savefig(output_path)
    print(f"Gráfico guardado en: {output_path}")
    plt.show()

if __name__ == "__main__":
    check_iam()
