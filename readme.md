# Simulación y Análisis de Señales con la Transformada de Fourier

Este repositorio contiene el desarrollo práctico de la **Actividad Formativa 2: Simulación y análisis de señales con la transformada de Fourier** implementado en Python. El proyecto analiza señales elementales en el dominio del tiempo y de la frecuencia aplicando la Transformada Rápida de Fourier (`np.fft.fft()`), grafica sus espectros de magnitud y fase, y verifica numérica y visualmente las propiedades fundamentales de la Transformada de Fourier.

## Propósito del Proyecto

El objetivo principal es comprender cómo la representación temporal de una señal se relaciona directamente con su contenido espectral en frecuencia:
1. Generar y visualizar tres señales elementales en el dominio del tiempo: **pulso rectangular**, **función escalón unitario (ventana causal acotada)** y **función senoidal pura**.
2. Calcular la Transformada Discreta de Fourier mediante el algoritmo FFT (`np.fft.fft`) en Python y extraer los espectros de **magnitud** $|X(f)|$ y **fase** $\angle X(f)$.
3. Demostrar experimentalmente las propiedades de **linealidad**, **desplazamiento en el tiempo** y **escalamiento en el tiempo y la frecuencia** (dualidad tiempo-frecuencia).

## Estructura del Repositorio

* `simulacion_fourier.py`: Script principal en Python ampliamente documentado con la definición matemática de las señales, cálculo de la FFT, verificación de propiedades e impresión de métricas de validación en consola.
* `informe_simulacion_fourier.pdf`: Documento técnico completo con el desarrollo de la actividad, marco teórico, gráficas de alta resolución, análisis comparativo tiempo-frecuencia, impacto de las propiedades y conclusiones.
* `presentacion_fourier.pptx`: Presentación ejecutiva en PowerPoint lista para la sesión en línea de exposición de hallazgos.
* `README.md`: Documentación general del repositorio, instrucciones de instalación, ejecución y flujo de trabajo con Git.

## Requisitos e Instalación

El proyecto utiliza Python 3.8+ junto con las librerías científicas estándar del ecosistema de procesamiento de señales:

| Librería | Propósito en la Simulación | Comando de Instalación |
| :--- | :--- | :--- |
| **NumPy** | Vectores temporales, operaciones matriciales y cálculo de `np.fft.fft()` / `np.fft.fftfreq()` | `pip install numpy` |
| **SciPy** | Herramientas complementarias de procesamiento de señales (`scipy.signal`) | `pip install scipy` |
| **Matplotlib** | Generación de gráficas de dominio del tiempo, magnitud y fase espectral | `pip install matplotlib` |

Para instalar todas las dependencias en un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install numpy scipy matplotlib
```

## Ejecución del Código

Para ejecutar la simulación, generar las gráficas del análisis y mostrar en consola el resumen de verificación numérica:

```bash
python3 simulacion_fourier.py
```

Al ejecutarse, el programa imprime la tabla de parámetros de muestreo ($f_s = 1000\text{ Hz}$, $N = 2000\text{ muestras}$, $\Delta f = 0.5\text{ Hz}$), verifica los cruces por cero y amplitudes teóricas, y exporta tres paneles gráficos de alta resolución:
* `chart_1_rect_pulse.png`: Pulso rectangular ($T_p = 0.20\text{ s}$) y su espectro tipo $\text{sinc}(f T_p)$.
* `chart_2_step_and_sine.png`: Escalón causal acotado ($T_u = 0.50\text{ s}$) y señal senoidal pura ($f_0 = 15\text{ Hz}$).
* `chart_3_fourier_properties.png`: Verificación gráfica de linealidad, desplazamiento temporal ($t_0 = 0.15\text{ s}$) y escalamiento tiempo-frecuencia ($a \in \{0.5, 1.0, 2.0\}$).

## Resumen de Resultados Clave

| Señal / Propiedad Analizada | Expresión en el Tiempo $x(t)$ | Representación en Frecuencia $X(f)$ | Hallazgo Experimental Clave |
| :--- | :--- | :--- | :--- |
| **Pulso Rectangular** | $\text{rect}(t / 0.20)$ | $0.20\,\text{sinc}(0.20 f)$ | Pico central $|X(0)| = 0.2000$ y cruces por cero exactos en múltiplos de $\pm 5.0\text{ Hz}$ ($1/T_p$). |
| **Escalón Causal Acotado** | $u(t) - u(t - 0.50)$ | $0.50\,\text{sinc}(0.50 f)\,e^{-j\pi f (0.50)}$ | Fuerte componente DC en $f = 0\text{ Hz}$ ($|U(0)| = 0.5000$) y fase lineal decreciente por asimetría causal. |
| **Función Senoidal** | $\sin(2\pi \cdot 15 t)$ | $\frac{1}{2j}[\delta(f - 15) - \delta(f + 15)]$ | Dos impulsos puros en $f = \pm 15.0\text{ Hz}$ con magnitud $0.5000$ y fases de $-90^\circ$ y $+90^\circ$. |
| **Linealidad** | $2\cos(2\pi \cdot 10 t) + 1.5\sin(2\pi \cdot 25 t)$ | $2 X_1(f) + 1.5 X_2(f)$ | Superposición exacta sin intermodulación; error numérico máximo de $3.14 \times 10^{-16}$. |
| **Desplazamiento Temporal** | $\text{rect}((t - 0.15)/0.20)$ | $X(f)\,e^{-j 2\pi f (0.15)}$ | Magnitud invariante ($\Delta |X(f)| = 0.00$) y rotación lineal de fase de $-54^\circ/\text{Hz}$ ($-108^\circ$ en $f = 2\text{ Hz}$). |
| **Escalamiento** | $\text{rect}(a t / 0.20)$ con $a \in \{0.5, 2.0\}$ | $\frac{1}{|a|} X(f/a)$ | Comprimir el tiempo a la mitad ($a=2$) duplica el ancho de banda del lóbulo principal (nulo pasa de $5\text{ Hz}$ a $10\text{ Hz}$). |

## Guía para Subir el Proyecto a GitHub

Para publicar estos archivos en tu cuenta de GitHub desde la terminal local, sigue esta secuencia de comandos:

```bash
# 1. Inicializar el repositorio Git local
git init

# 2. Configurar rama principal como 'main'
git branch -M main

# 3. Agregar el script de simulación y la documentación
git add simulacion_fourier.py README.md

# 4. Registrar el primer commit descriptivo
git commit -m "feat: implementar simulación de Transformada de Fourier y análisis de propiedades en Python"

# 5. Vincular con el repositorio remoto creado en GitHub (reemplaza <TU_USUARIO> con tu cuenta)
git remote add origin https://github.com/<TU_USUARIO>/simulacion-transformada-fourier.git

# 6. Subir los cambios al repositorio en línea
git push -u origin main
```
