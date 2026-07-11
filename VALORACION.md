# Valoración de la estrategia intradía de IAG (`iag_intraday_event_study.py`)

**Fecha:** 2026-07-11 · **Activo:** IAG (BME, `IAG.MC`) · **Datos:** velas 5 min, ~60 días vía yfinance

## 1. Qué hace la estrategia

Es un **estudio de eventos** de reversión intradía, no una simple regla mecánica:

> Si IAG ha **subido X %** desde la apertura hasta una hora de control (09:15 / 09:30 / 10:00),
> ¿tiende a **caer** después hasta una hora de salida fija (12:00 / 15:00 / 17:25)?

- **Señal:** retorno desde `day_open` hasta el cierre de la última vela **ya cerrada** antes del *checkpoint*.
- **Entrada:** apertura de la vela siguiente al *checkpoint* (posición **corta**).
- **Salida:** cierre de la última vela cerrada antes de la hora de salida.
- **Malla explorada:** 3 *checkpoints* × 4 umbrales (0,5 / 1,0 / 1,5 / 2,0 %) × 3 salidas = **36 configuraciones**.
- **Métricas:** retorno bruto y neto (10 bps ida+vuelta), *win rate*, t-stat, IC 95 % por *bootstrap*,
  retorno compuesto, MFE/MAE, y comparación contra la media de *todos* los días (`edge_vs_all_days`).

## 2. Veredicto

**El código es correcto y honesto; la lógica hace exactamente lo que dice.** Lo que aún **no** es
—y el propio autor lo reconoce en las instrucciones— es una estrategia *validada*. Con 60 días y 36
celdas es una **exploración**, no una prueba. Los puntos débiles no son bugs, son de **metodología,
estadística y realismo de ejecución** (sobre todo en el lado corto).

### Verificación realizada
No pude descargar datos reales en este entorno (el proxy deniega Yahoo Finance, `fc.yahoo.com` → 403),
así que validé la **lógica** con datos sintéticos (`tests/synthetic_check.py`). Resultado:

- Pipeline completo, 45 sesiones, 405 observaciones. ✅
- **Sin look-ahead:** 0 filas con `salida ≤ entrada`; la señal usa solo velas cerradas y la entrada es
  la apertura de la vela siguiente. ✅
- **Signo correcto:** inyectando un patrón "sube por la mañana → revierte por la tarde", el corto sale
  ganador (media +1,64 %, *edge* +1,04 %, t≈19). La dirección de la apuesta y la estadística están bien. ✅

## 3. Lo que está bien hecho

1. **Sin look-ahead.** La distinción `closed_bars_before` (señal) vs `entry_bar_at_or_after` (entrada) es
   correcta y es el error nº 1 en backtests intradía. Aquí está evitado.
2. **Entrada en la apertura de la vela siguiente,** no en el cierre de la vela de señal. Esto evita el
   artefacto de *bid-ask bounce* (la reversión espuria que aparece si entras al mismo precio con el que
   mides la señal). Buena decisión.
3. **Costes explícitos** (10 bps ida+vuelta) y parametrizables por CLI.
4. **Comparación contra línea base** (`edge_vs_all_days`): distingue el retorno condicional del
   incondicional, imprescindible al operar en corto un valor con deriva alcista.
5. **Inferencia:** t-stat, IC por *bootstrap* por bloques (eficiente en memoria), *win rate*, MFE/MAE.
6. **Higiene de datos:** normalización de columnas multiíndice de yfinance, zona horaria Europe/Madrid,
   filtrado de sesiones incompletas (incluido el día en curso).
7. **Transparencia:** las instrucciones ya advierten de que 60 días son exploratorios y de que faltan
   spread variable, préstamo de acciones y *slippage*.

## 4. Debilidades y riesgos (ordenados por severidad)

| # | Severidad | Problema | Impacto |
|---|-----------|----------|---------|
| 1 | **Alta** | **Techo de datos.** yfinance solo da velas 5 min de ~60 días. Con esta fuente es *imposible* validar multi-año u out-of-sample. | La hipótesis nunca podrá confirmarse tal cual. |
| 2 | **Alta** | **Multiplicidad / sobreajuste.** 36 celdas, sin corrección por comparaciones múltiples. Elegir la mejor celda ≈ minería de datos. | *Win rate* y t-stat "buenos" pueden ser ruido. |
| 3 | **Alta** | **Muestra pequeña.** Con umbral 1,5–2 % los eventos caen a un puñado (a veces 0, como se ve en la salida). El *bootstrap* con n<10 no es fiable. | Conclusiones frágiles en las celdas "interesantes". |
| 4 | **Alta** | **Realismo del corto no modelado.** Coste/disponibilidad de préstamo, riesgo de veto de la CNMV a cortos, y cruce de spread. 10 bps es optimista para un corto retail. | El *edge* sobre el papel puede desaparecer en real. |
| 5 | **Media** | **Sin stop-loss.** Se mantiene el corto hasta una hora fija; el MAE se registra pero no se usa. Un *short squeeze* tiene cola de pérdida no acotada. | Riesgo de ruina no controlado. |
| 6 | **Media** | **La significación se mide contra 0, no contra la línea base.** El t-stat/IC prueban "media ≠ 0", pero lo relevante es "evento ≠ no-evento". `edge_vs_all_days` se calcula pero no se contrasta. | Puede declarar significativo algo que solo refleja la deriva del valor. |
| 7 | **Media** | **Línea base contaminada:** `baseline` = media de *todos* los días (los eventos ⊂ base). Lo limpio es evento vs no-evento. | Sesga a la baja el *edge* aparente. |
| 8 | **Baja** | Faltan **Sharpe, drawdown máximo, profit factor** y anualización. `compounded_return` asume 100 % del capital por operación. | Difícil juzgar riesgo/retorno real. |
| 9 | **Baja** | `gap_pct` usa el cierre de la sesión previa *procesada*; si se saltó un día incompleto, no es contigua. No afecta a la estrategia (solo se registra). | Cosmético. |
| 10 | **Baja** | Sin manejo de *splits*/dividendos (`actions=False`). Improbable en 60 días, pero un split rompería la serie. | Marginal en esta ventana. |

## 5. Recomendaciones concretas

**Para poder validar (imprescindible):**
- Conseguir **datos intradía de varios años** de otra fuente (proveedor de pago, `IAG.L` en Londres con
  otro feed, o datos de tu bróker). Reservar el **último tramo como out-of-sample** y no tocarlo hasta el final.
- **Fijar una sola hipótesis** (p. ej. checkpoint 09:30, umbral 1 %, salida 17:25) *antes* de mirar
  resultados; usar el resto de celdas solo como robustez, no para elegir la ganadora.

**Estadística:**
- Añadir un **test evento vs no-evento** (diferencia de medias / permutación) y reportar su p-valor.
- Corrección por multiplicidad (Bonferroni/BH) si se sigue explorando la malla.
- Exigir un **n mínimo** (p. ej. ≥30 eventos) antes de dar por buena una celda.

**Realismo:**
- Modelar **spread** (cruzarlo en entrada y salida), **slippage** y **coste de préstamo** diario del corto.
- Incluir **stop-loss** basado en el MAE observado y recalcular; comparar con la salida a hora fija.
- Métricas de riesgo: **Sharpe, drawdown máximo, profit factor**, retorno por operación.

**Robustez:**
- *Walk-forward* / validación en ventanas móviles.
- Comprobar la simetría (¿funciona igual el lado largo tras caídas matinales?).
- Análisis de sensibilidad a los parámetros (que el resultado no dependa de un umbral exacto).

## 6. Cómo ejecutar

```bash
pip install -r requirements.txt
python iag_intraday_event_study.py                 # descarga real (necesita red a Yahoo)
python iag_intraday_event_study.py --cost-bps 20   # coste más conservador
python tests/synthetic_check.py                    # verificación de lógica, sin red
```

## 7. Mejoras ya aplicadas al código

Sobre la versión original se han implementado (verificadas con `tests/synthetic_check.py`):

| Antes | Ahora |
|-------|-------|
| Coste único de 10 bps | **Comisión + spread cruzado + préstamo prorrateado** (`--spread-bps`, `--borrow-annual-pct`). |
| Sin stop; pérdida no acotada | **Stop-loss** configurable del corto (`--stop-pct`), con recuento de salidas por stop. |
| Significación solo contra 0 | **Contraste evento vs no-evento por permutación** (`perm_pvalue`). |
| Sin métricas de riesgo | **Sharpe por operación, drawdown máximo, profit factor**. |
| Datos atados a yfinance (~60 días) | **Lector de CSV propio** (`--csv`) para validar con años de historia. |
| — | Aviso explícito de multiplicidad (36 configuraciones) en la salida. |

Pendiente (decisión metodológica, no de código): conseguir años de datos reales,
**fijar una sola hipótesis de antemano**, reservar un tramo out-of-sample intocado y
hacer *walk-forward*. El código ya está preparado para todo ello vía `--csv`.

## 8. Conclusión

Buen punto de partida: **infraestructura limpia, sin look-ahead y con la dirección correcta.** La barrera
real no es el código, son los **datos (60 días es poco), el sobreajuste de 36 celdas y el realismo del
corto**. Antes de arriesgar capital: más historia, una hipótesis fijada de antemano, out-of-sample
intocado, costes de préstamo/spread y un stop. Trátalo como un experimento prometedor, no como una señal
lista para operar.
