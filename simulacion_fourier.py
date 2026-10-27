#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Actividad Formativa 2: Simulación y Análisis de Señales con la Transformada de Fourier
================================================================================
Descripción:
    Este script implementa en Python la generación, simulación y análisis de
    señales elementales en el dominio del tiempo y de la frecuencia utilizando
    la Transformada Rápida de Fourier (FFT) mediante `np.fft.fft()`.

    Estructura del análisis:
    1. Definición de señales elementales en el dominio del tiempo:
       - Pulso rectangular: rect(t / T_p)
       - Función escalón unitario (ventana finita causal): u(t)
       - Función senoidal pura: A * sin(2 * pi * f_0 * t)
    2. Cálculo de la Transformada de Fourier (magnitud y fase).
    3. Visualización gráfica de las señales y sus espectros de frecuencia.
    4. Análisis y verificación numérica y gráfica de propiedades fundamentales:
       - Propiedad de Linealidad: F{a*x1(t) + b*x2(t)} = a*X1(f) + b*X2(f)
       - Propiedad de Desplazamiento en el Tiempo: F{x(t - t0)} = X(f)*exp(-j*2*pi*f*t0)
       - Propiedad de Escalamiento en el Tiempo/Frecuencia: F{x(a*t)} = (1/|a|)*X(f/a)

Autor: Desarrollo de Simulación en Python (NumPy, SciPy, Matplotlib)
================================================================================
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Configuración de estilo visual para gráficas científicas de alta resolución
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
    "grid.color": "#CBD5E1",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

# Paleta semántica coherente
COLOR_PRIMARY = "#0F172A"    # Azul marino profundo
COLOR_ACCENT = "#0284C7"     # Azul cian técnico
COLOR_SECONDARY = "#D97706"  # Ámbar / naranja para contraste y fase
COLOR_TERTIARY = "#059669"   # Verde esmeralda para verificación

# Directorio de salida para almacenar las figuras generadas
OUTPUT_FIG_DIR = os.environ.get("FIG_OUTPUT_DIR", "graficas")
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# ==============================================================================
# 1. DEFINICIÓN DE PARÁMETROS Y SEÑALES EN EL DOMINIO DEL TIEMPO
# ==============================================================================
# Parámetros de muestreo
fs = 1000.0                  # Frecuencia de muestreo (Hz)
dt = 1.0 / fs                # Periodo de muestreo (s) -> 1 ms
T_total = 2.0                # Ventana total de observación (s): de -1.0 s a +1.0 s
N = int(T_total * fs)        # Número total de muestras (N = 2000)
t = np.linspace(-1.0, 1.0, N, endpoint=False)  # Vector de tiempo centrado en t = 0

# 1.1 Pulso Rectangular: ancho T_p = 0.20 s centrado en el origen (|t| <= 0.10 s)
T_p = 0.20
x_rect = np.where(np.abs(t) <= (T_p / 2.0), 1.0, 0.0)

# 1.2 Función Escalón Unitario Causal Acotado: u(t) activa en [0, 0.50 s]
# Nota: En simulación discreta con FFT se acota en ventana finita T_u = 0.50 s
# para garantizar integrabilidad absoluta (condición de Dirichlet) y evitar aliasing severo.
T_u = 0.50
x_step = np.where((t >= 0.0) & (t <= T_u), 1.0, 0.0)

# 1.3 Función Senoidal Pura: f0 = 15 Hz, Amplitud A = 1.0 V
f0 = 15.0
A_sin = 1.0
x_sin = A_sin * np.sin(2.0 * np.pi * f0 * t)


# ==============================================================================
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER (USANDO np.fft.fft)
# ==============================================================================
def calcular_espectro_fft(x, t_vec, fs_val, umbral_fase=0.05):
    """
    Calcula la Transformada de Fourier centrada en cero utilizando np.fft.fft().

    Parámetros:
        x (np.ndarray): Señal en el dominio del tiempo.
        t_vec (np.ndarray): Vector de tiempo correspondiente.
        fs_val (float): Frecuencia de muestreo en Hz.
        umbral_fase (float): Fracción de la magnitud máxima por debajo de la cual
                             se enmascara la fase para eliminar ruido numérico.

    Retorna:
        f (np.ndarray): Vector de frecuencias en Hz centrado en 0 Hz.
        X_complex (np.ndarray): Espectro complejo continuo aproximado (con corrección de fase temporal).
        magnitud (np.ndarray): Magnitud del espectro |X(f)|.
        fase_grados (np.ndarray): Fase del espectro en grados (enmascarada donde |X(f)| es despreciable).
    """
    n_samples = len(x)
    dt_val = 1.0 / fs_val

    # Cálculo directo de la DFT mediante np.fft.fft() y centrado con fftshift
    X_raw = np.fft.fft(x)
    f = np.fft.fftshift(np.fft.fftfreq(n_samples, d=dt_val))
    X_shifted = np.fft.fftshift(X_raw) * dt_val

    # Corrección de fase por el origen del vector temporal (t[0] = -1.0 s)
    # para alinear la fase de la DFT con la integral continua de Fourier
    t0_offset = t_vec[0]
    factor_fase_origen = np.exp(-1j * 2.0 * np.pi * f * t0_offset)
    X_complex = X_shifted * factor_fase_origen

    magnitud = np.abs(X_complex)

    # Enmascaramiento de fase donde la magnitud es cercana a cero (evita ruido aleatorio)
    umbral = umbral_fase * np.max(magnitud)
    fase_rad = np.where(magnitud >= umbral, np.angle(X_complex), 0.0)
    fase_grados = np.degrees(fase_rad)

    return f, X_complex, magnitud, fase_grados


# Cálculo del espectro para cada señal elemental
f_vec, X_rect, mag_rect, fase_rect = calcular_espectro_fft(x_rect, t, fs, umbral_fase=0.02)
_, X_step, mag_step, fase_step = calcular_espectro_fft(x_step, t, fs, umbral_fase=0.02)
# Para la senoidal normalizamos por N/2 para visualizar la amplitud de pico en frecuencia
X_sin_raw = np.fft.fftshift(np.fft.fft(x_sin)) / N
factor_origen = np.exp(-1j * 2.0 * np.pi * f_vec * t[0])
X_sin = X_sin_raw * factor_origen
mag_sin = np.abs(X_sin)
fase_sin = np.where(mag_sin >= 0.1 * np.max(mag_sin), np.degrees(np.angle(X_sin)), 0.0)


# ==============================================================================
# 3. VISUALIZACIÓN DE LOS RESULTADOS DE LAS SEÑALES ELEMENTALES
# ==============================================================================
# Figura 1: Pulso Rectangular (Tiempo, Magnitud Sinc, Fase)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
axes[0].plot(t, x_rect, color=COLOR_PRIMARY, lw=2.0)
axes[0].fill_between(t, 0, x_rect, color=COLOR_ACCENT, alpha=0.18)
axes[0].set_xlim(-0.5, 0.5)
axes[0].set_ylim(-0.15, 1.25)
axes[0].set_title("a) Dominio del Tiempo: Pulso Rectangular")
axes[0].set_xlabel("Tiempo t (s)")
axes[0].set_ylabel("Amplitud x(t)")
axes[0].grid(True)

mask_f50 = np.abs(f_vec) <= 30.0
axes[1].plot(f_vec[mask_f50], mag_rect[mask_f50], color=COLOR_ACCENT, lw=2.0, label="FFT Numérica")
# Curva teórica: T_p * |sinc(f * T_p)|
mag_teorica_rect = T_p * np.abs(np.sinc(f_vec[mask_f50] * T_p))
axes[1].plot(f_vec[mask_f50], mag_teorica_rect, "--", color=COLOR_SECONDARY, lw=1.4, label="Teórica |T·sinc(fT)|")
axes[1].set_title("b) Espectro de Magnitud |X(f)|")
axes[1].set_xlabel("Frecuencia f (Hz)")
axes[1].set_ylabel("Magnitud |X(f)|")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True)

axes[2].plot(f_vec[mask_f50], fase_rect[mask_f50], color=COLOR_SECONDARY, lw=1.6)
axes[2].set_ylim(-200, 200)
axes[2].set_title("c) Espectro de Fase ∠X(f)")
axes[2].set_xlabel("Frecuencia f (Hz)")
axes[2].set_ylabel("Fase (grados)")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "chart_1_rect_pulse.png"))
plt.close()

# Figura 2: Función Escalón Causal Acotada y Función Senoidal
fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8))

# Fila 1: Escalón unitario acotado u(t) - u(t - 0.5)
axes[0, 0].plot(t, x_step, color=COLOR_PRIMARY, lw=2.0)
axes[0, 0].fill_between(t, 0, x_step, color=COLOR_ACCENT, alpha=0.18)
axes[0, 0].set_xlim(-0.4, 0.8)
axes[0, 0].set_ylim(-0.15, 1.25)
axes[0, 0].set_title("a) Escalón Causal Acotado u(t) [0, 0.5 s]")
axes[0, 0].set_xlabel("Tiempo t (s)")
axes[0, 0].set_ylabel("Amplitud")
axes[0, 0].grid(True)

mask_f20 = np.abs(f_vec) <= 15.0
axes[0, 1].plot(f_vec[mask_f20], mag_step[mask_f20], color=COLOR_ACCENT, lw=2.0)
axes[0, 1].set_title("b) Magnitud |U(f)| (Pico DC en f = 0 Hz)")
axes[0, 1].set_xlabel("Frecuencia f (Hz)")
axes[0, 1].set_ylabel("Magnitud |U(f)|")
axes[0, 1].grid(True)

axes[0, 2].plot(f_vec[mask_f20], fase_step[mask_f20], color=COLOR_SECONDARY, lw=1.6)
axes[0, 2].set_ylim(-200, 200)
axes[0, 2].set_title("c) Fase ∠U(f) (Rampa lineal por asimetría causal)")
axes[0, 2].set_xlabel("Frecuencia f (Hz)")
axes[0, 2].set_ylabel("Fase (grados)")
axes[0, 2].grid(True)

# Fila 2: Función Senoidal de f0 = 15 Hz
axes[1, 0].plot(t, x_sin, color=COLOR_PRIMARY, lw=1.6)
axes[1, 0].set_xlim(-0.2, 0.2)
axes[1, 0].set_ylim(-1.3, 1.3)
axes[1, 0].set_title("d) Señal Senoidal Pura (f0 = 15 Hz)")
axes[1, 0].set_xlabel("Tiempo t (s)")
axes[1, 0].set_ylabel("Amplitud")
axes[1, 0].grid(True)

mask_f30 = np.abs(f_vec) <= 30.0
axes[1, 1].stem(
    f_vec[mask_f30], mag_sin[mask_f30],
    linefmt=COLOR_ACCENT, markerfmt="o", basefmt=" "
)
axes[1, 1].set_ylim(0, 0.65)
axes[1, 1].set_title("e) Magnitud |S(f)| (Impulsos en ±15 Hz = 0.5)")
axes[1, 1].set_xlabel("Frecuencia f (Hz)")
axes[1, 1].set_ylabel("Magnitud Normalizada")
axes[1, 1].grid(True)

axes[1, 2].stem(
    f_vec[mask_f30], fase_sin[mask_f30],
    linefmt=COLOR_SECONDARY, markerfmt="s", basefmt=" "
)
axes[1, 2].set_ylim(-120, 120)
axes[1, 2].set_title("f) Fase ∠S(f) (-90° en +15 Hz, +90° en -15 Hz)")
axes[1, 2].set_xlabel("Frecuencia f (Hz)")
axes[1, 2].set_ylabel("Fase (grados)")
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "chart_2_step_and_sine.png"))
plt.close()


# ==============================================================================
# 4. ANÁLISIS DE PROPIEDADES Y TRANSFORMACIONES DE LA TRANSFORMADA DE FOURIER
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.1 Propiedad de Linealidad: F{a*x1(t) + b*x2(t)} = a*X1(f) + b*X2(f)
# ------------------------------------------------------------------------------
a_coef = 2.0
b_coef = 1.5
f1_lin = 10.0  # Hz
f2_lin = 25.0  # Hz

x1_lin = np.cos(2.0 * np.pi * f1_lin * t)
x2_lin = np.sin(2.0 * np.pi * f2_lin * t)
x_comb = a_coef * x1_lin + b_coef * x2_lin

# Transformada directa de la señal combinada vs suma ponderada de transformadas individuales
X1_fft = np.fft.fftshift(np.fft.fft(x1_lin)) / N
X2_fft = np.fft.fftshift(np.fft.fft(x2_lin)) / N
X_comb_directa = np.fft.fftshift(np.fft.fft(x_comb)) / N
X_comb_suma = a_coef * X1_fft + b_coef * X2_fft

error_linealidad_max = np.max(np.abs(X_comb_directa - X_comb_suma))

# ------------------------------------------------------------------------------
# 4.2 Propiedad de Desplazamiento en el Tiempo: F{x(t - t0)} = X(f)*e^(-j*2*pi*f*t0)
# ------------------------------------------------------------------------------
t0_shift = 0.15  # Retardo de 150 ms
x_rect_shifted = np.where(np.abs(t - t0_shift) <= (T_p / 2.0), 1.0, 0.0)
_, X_rect_shift, mag_rect_shift, fase_rect_shift = calcular_espectro_fft(
    x_rect_shifted, t, fs, umbral_fase=0.02
)

# Diferencia entre magnitudes (debe ser ~0: el desplazamiento temporal no altera la magnitud)
diff_magnitud_shift = np.max(np.abs(mag_rect - mag_rect_shift))

# ------------------------------------------------------------------------------
# 4.3 Propiedad de Escalamiento en Tiempo/Frecuencia: F{x(a*t)} = (1/|a|)*X(f/a)
# ------------------------------------------------------------------------------
# Compresión temporal con factor a = 2.0 (ancho del pulso se reduce a la mitad: T_p/2 = 0.10 s)
a_scale = 2.0
x_rect_compressed = np.where(np.abs(a_scale * t) <= (T_p / 2.0), 1.0, 0.0)
_, X_rect_comp, mag_rect_comp, _ = calcular_espectro_fft(
    x_rect_compressed, t, fs, umbral_fase=0.02
)

# Expansión temporal con factor a = 0.5 (ancho del pulso se duplica: 2*T_p = 0.40 s)
a_expand = 0.5
x_rect_expanded = np.where(np.abs(a_expand * t) <= (T_p / 2.0), 1.0, 0.0)
_, X_rect_exp, mag_rect_exp, _ = calcular_espectro_fft(
    x_rect_expanded, t, fs, umbral_fase=0.02
)

# Figura 3: Visualización de las 3 Propiedades Fundamentales
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

# Panel a: Linealidad
mask_f40 = (f_vec >= 0.0) & (f_vec <= 40.0)
axes[0].plot(
    f_vec[mask_f40], 2.0 * np.abs(X_comb_directa[mask_f40]),
    color=COLOR_PRIMARY, lw=2.2, label="Directa |F{a·x1 + b·x2}|"
)
axes[0].plot(
    f_vec[mask_f40], 2.0 * np.abs(X_comb_suma[mask_f40]),
    "--", color=COLOR_SECONDARY, lw=1.8, label="Suma |a·X1(f) + b·X2(f)|"
)
axes[0].set_title("a) Propiedad de Linealidad (10 Hz y 25 Hz)")
axes[0].set_xlabel("Frecuencia f (Hz)")
axes[0].set_ylabel("Amplitud Espectral Unilateral")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True)

# Panel b: Desplazamiento en el Tiempo (Magnitud invariante + pendiente de fase)
mask_f15 = np.abs(f_vec) <= 10.0
axes[1].plot(
    f_vec[mask_f15], mag_rect[mask_f15],
    color=COLOR_PRIMARY, lw=2.2, label="Original |X(f)| (t0 = 0 s)"
)
axes[1].plot(
    f_vec[mask_f15], mag_rect_shift[mask_f15],
    "--", color=COLOR_SECONDARY, lw=1.8, label="Desplazada |X_d(f)| (t0 = 0.15 s)"
)
ax_phase = axes[1].twinx()
mask_lobe = np.abs(f_vec) <= 4.5  # Lóbulo principal donde no hay inversión de signo del sinc
ax_phase.plot(
    f_vec[mask_lobe], fase_rect_shift[mask_lobe],
    ":", color=COLOR_TERTIARY, lw=1.8, label="Fase ∠X_d(f) (-360°·f·t0)"
)
ax_phase.set_ylabel("Fase Lóbulo Principal (°)", color=COLOR_TERTIARY, fontsize=8.5)
ax_phase.tick_params(axis="y", labelcolor=COLOR_TERTIARY)
ax_phase.set_ylim(-260, 260)
axes[1].set_title("b) Desplazamiento Temporal (t0 = 0.15 s)")
axes[1].set_xlabel("Frecuencia f (Hz)")
axes[1].set_ylabel("Magnitud |X(f)|")
axes[1].legend(loc="upper left", fontsize=7.5)
axes[1].grid(True)

# Panel c: Escalamiento Tiempo-Frecuencia
mask_f25 = np.abs(f_vec) <= 20.0
axes[2].plot(
    f_vec[mask_f25], mag_rect_exp[mask_f25],
    color=COLOR_TERTIARY, lw=1.8, label="Expandida (a=0.5, Tp=0.4s)"
)
axes[2].plot(
    f_vec[mask_f25], mag_rect[mask_f25],
    color=COLOR_PRIMARY, lw=2.0, label="Base (a=1.0, Tp=0.2s)"
)
axes[2].plot(
    f_vec[mask_f25], mag_rect_comp[mask_f25],
    color=COLOR_ACCENT, lw=2.0, label="Comprimida (a=2.0, Tp=0.1s)"
)
axes[2].set_title("c) Escalamiento Tiempo-Frecuencia")
axes[2].set_xlabel("Frecuencia f (Hz)")
axes[2].set_ylabel("Magnitud |X(f)|")
axes[2].legend(loc="upper right", fontsize=8)
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "chart_3_fourier_properties.png"))
plt.close()


# ==============================================================================
# 5. REPORTE DE VERIFICACIÓN NUMÉRICA EN CONSOLA
# ==============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("RESUMEN DE SIMULACIÓN Y VERIFICACIÓN DE LA TRANSFORMADA DE FOURIER")
    print("=" * 72)
    print(f"Frecuencia de muestreo (fs):         {fs:.1f} Hz (dt = {dt*1000:.2f} ms)")
    print(f"Muestras totales (N):                {N} muestras (Ventana = {T_total:.1f} s)")
    print(f"Resolución espectral (df = fs/N):    {fs/N:.2f} Hz")
    print("-" * 72)
    print("1. SEÑALES ELEMENTALES:")
    print(f"   - Pulso Rectangular (Tp = {T_p:.2f} s):   Pico en f=0 Hz = {np.max(mag_rect):.4f} (Teórico = {T_p:.4f})")
    print(f"     Primer cruce por cero espectral:   f = ±{1.0/T_p:.2f} Hz (1/Tp)")
    print(f"   - Escalón Acotado (Tu = {T_u:.2f} s):     Pico DC en f=0 Hz = {np.max(mag_step):.4f}")
    print(f"   - Senoidal Pura (f0 = {f0:.1f} Hz):       Picos en ±{f0:.1f} Hz con magnitud = {np.max(mag_sin):.4f} (A/2 = 0.5000)")
    print("-" * 72)
    print("2. VERIFICACIÓN DE PROPIEDADES:")
    print(f"   - Linealidad Error Máx |F{{ax1+bx2}} - (aX1+bX2)|: {error_linealidad_max:.3e}")
    print(f"   - Desplazamiento Temporal (t0 = {t0_shift:.2f} s):")
    print(f"     Diferencia máxima de magnitud |X(f)| vs |X_d(f)|: {diff_magnitud_shift:.3e}")
    print(f"     Pendiente de fase medida en f = 2 Hz:             {fase_rect_shift[np.argmin(np.abs(f_vec - 2.0))]:.2f}° (Teórica = {-360.0*2.0*t0_shift:.2f}°)")
    print(f"   - Escalamiento Temporal:")
    print(f"     a = 0.5 (Tp=0.4s) -> Pico |X(0)| = {np.max(mag_rect_exp):.4f}, 1er nulo = ±{1.0/0.4:.2f} Hz")
    print(f"     a = 1.0 (Tp=0.2s) -> Pico |X(0)| = {np.max(mag_rect):.4f}, 1er nulo = ±{1.0/0.2:.2f} Hz")
    print(f"     a = 2.0 (Tp=0.1s) -> Pico |X(0)| = {np.max(mag_rect_comp):.4f}, 1er nulo = ±{1.0/0.1:.2f} Hz")
    print("=" * 72)