import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime



sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12,6)

# Se obtienen los datos de internet
simbolo = "^MXX" # Este es el símbolo del IPC en Yahoo Finanzas
fecha_inicio = "2019-01-01"
fecha_termino = "2024-12-31"


# Descarga
print(f"Se descargan los datos del IPC desde {fecha_inicio} hasta {fecha_termino}")
datos_ipc = yf.download(simbolo, start=fecha_inicio, end=fecha_termino)


# Se observan las primeras filas
#print("\nPrimeras filas de datos:")
#print(datos_ipc.head())

# Datos generales
#print("\nInformación general del dataset:")
#print(datos_ipc.info())

# Se añaden solo los precios ajustados al cierre
IPC = datos_ipc['Close'].copy()

#print("\nPrimeras filas de IPC:")
#print(IPC.head())

# Verificación de valores faltantes
#print(f"\nValores faltantes: {IPC.isna().sum()}")

# En este caso, dado que no existen NAs, es posible proceder sin complicaciones. En caso opuesto, sería recomendable
# utilizar un modelo ARIMA que maneje internamente los NAs, o incluso un Kalman filter para evitar
# autocorrelaciones espurias que podrián traeer problemas graves en series volátiles, particularmente las que provienen
# de procesos no-markovianos.

#print("\nEstadísticas descriptivas del IPC:")
#print(IPC.head(10))
#print(IPC.describe())

# Se crea la figura
plt.figure(figsize=(12,6))

# Se grafica la serie de precios
plt.plot(IPC.index, IPC.values, linewidth=2, color='darkblue', label='IPC')

# Se agregan títulos y etiquetas
plt.title('Índice de precios y cotizaciones (IPC) - Bolsa Mexicana de Valores', fontsize=16, fontweight='bold')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precios', fontsize=12)
plt.legend(loc='upper left', fontsize=10)

# Otros ajustes sencillos
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()


# Gráfico
plt.savefig('ipc_serie_temporal.png', dpi=300, bbox_inches='tight')
print("Gráfica guardada como 'ipc_serie_temporal.png'")
plt.close()



# Se trabaja ahora con los rendimientos logarítmicos
Rendimientos = np.log(IPC/IPC.shift(1))
Rendimientos = Rendimientos.dropna()

# Estadísticos y pruebas iniciales
print("\nEstadísticas de los rendimientos logarítmicos:")
print(Rendimientos.describe())
print(f"\nNúmero de observaciones: {len(Rendimientos)}")
print("\nPrimeros rendimientos:")
print(Rendimientos.head(10))

# Figura con dos subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Gráfica 1: Serie de tiempo de rendimientos
axes[0].plot(Rendimientos.index, Rendimientos.values, linewidth=0.8, color='darkgreen')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[0].set_title('Rendimientos Logarítmicos Diarios del IPC', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fecha', fontsize=11)
axes[0].set_ylabel('Rendimiento', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Gráfica 2: Histograma de rendimientos

media_rendimientos = Rendimientos.values.mean()

axes[1].hist(Rendimientos, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=media_rendimientos, color='red', linestyle='--', linewidth=2, label=f'Media = {media_rendimientos:.6f}')
axes[1].set_title('Distribución de Rendimientos Logarítmicos', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Rendimiento', fontsize=11)
axes[1].set_ylabel('Frecuencia', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Ajustar espaciado entre gráficas
plt.tight_layout()

# Guardar la figura
plt.savefig('rendimientos_ipc.png', dpi=300, bbox_inches='tight')
print("Gráfica guardada como 'rendimientos_ipc.png'")
plt.close()

# Se importan ahora las bibliotecas para las pruebas estadísticas
from statsmodels.tsa.stattools import pacf, adfuller, acf
from scipy import stats


print("\n" + "="*60)
print("Se usa la prueba de estacionalidad  Dickey-Fuller (ADF)")
print("="*60)

resultado_adf = adfuller(Rendimientos, autolag='AIC')

# Se obtienen resultados de esto
estadisticas_adf = resultado_adf[0]
valor_p = resultado_adf[1]
n_rezagos = resultado_adf[2]
n_obs = resultado_adf[3]
valores_criticos = resultado_adf[4]

# Se muestran estos resultados
print(f"\nEstadístico ADF: {estadisticas_adf:.6f}")
print(f"Valor P: {valor_p:.10f}")
print(f"Número de rezagos: {n_rezagos:.6f}")
print(f"Número de observaciones: {n_obs:.6f}")
print(f"Valores críticos:")
for clave, valor in valores_criticos.items():
    print(f"{clave}: {valor:.6f}")

# Interpretación de estos resultados, si el valor P es mayor que 0.05 no se rechaza
if valor_p < 0.05:
    print("Se rechaza H0: La serie es estacionaria")
else:
    print("No se rechaza H0: La serie no es estacionaria")







# Autocorrleación con ACF y PACF

print("\n" + "="*60)
print("Análisis de autocorrelación")
print("="*60)

# Calcular ACF y PACF
valores_acf = acf(Rendimientos, nlags=40, fft=False)
valores_pacf = pacf(Rendimientos, nlags=40, method='ywm')

# Crear figura con dos gráficas
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Gráfica ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(Rendimientos, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('Función de Autocorrelación (ACF)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rezagos', fontsize=11)
axes[0].set_ylabel('Autocorrelación', fontsize=11)

# Gráfica PACF
plot_pacf(Rendimientos, lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('Función de Autocorrelación Parcial (PACF)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Rezagos', fontsize=11)
axes[1].set_ylabel('Autocorrelación Parcial', fontsize=11)

plt.tight_layout()
plt.savefig('acf_pacf_rendimientos.png', dpi=300, bbox_inches='tight')
print("\nGráfica ACF/PACF guardada como 'acf_pacf_rendimientos.png'")
plt.close()

# Mostrar algunos valores de ACF y PACF
print("\nPrimeros 10 valores de ACF:")
print(valores_acf[:10])
print("\nPrimeros 10 valores de PACF:")
print(valores_pacf[:10])
