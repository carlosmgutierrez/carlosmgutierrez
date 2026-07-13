# Estudio intradía de IAG (BME)

Backtest exploratorio de un patrón de **reversión intradía** en IAG (`IAG.MC`): tras una subida
matinal, ¿tiende el valor a caer hasta el cierre? Posición **corta**, velas de 5 minutos.

## Contenido

| Fichero | Descripción |
|---------|-------------|
| `iag_intraday_event_study.py` | Estudio de eventos: descarga, señal sin look-ahead, entrada/salida, estadística y figuras. |
| `fetch_iag_data.py` | Acumulador incremental: descarga los ~60 días de Yahoo y los fusiona sin duplicar en un CSV creciente. |
| `descargar_60d.py` | Lanzador de un solo paso: descarga 60 días de velas 5m y ejecuta el análisis completo. Requiere red a Yahoo. |
| `iag_deep_scan.py` | Exploración profunda: ~950 combinaciones de corto intradía (hora/subida fija → hora/objetivo), con prueba out-of-sample anti-sobreajuste y figuras. |
| `INSTRUCCIONES_IAG_INTRADIA.txt` | Instrucciones de uso. |
| `VALORACION.md` | **Informe de valoración** de la estrategia (fortalezas, debilidades y recomendaciones). |
| `tests/synthetic_check.py` | Verificación de la lógica con datos sintéticos (no necesita red). |
| `requirements.txt` | Dependencias. |

## Uso rápido

```bash
pip install -r requirements.txt
python iag_intraday_event_study.py                 # descarga de Yahoo (necesita red)
python iag_intraday_event_study.py --csv datos.csv # datos propios de varios años
python tests/synthetic_check.py                    # comprueba la lógica sin red
```

Los resultados se guardan en `iag_intraday_results/` (CSV + PNG).

### Conseguir historia intradía

Yahoo solo da ~60 días de velas de 5 min. Para acumular más historia, ejecuta el acumulador
periódicamente (p. ej. semanalmente); fusiona la ventana rodante en un CSV creciente sin duplicar:

```bash
python fetch_iag_data.py            # crea/actualiza iag_history_5m.csv
python iag_intraday_event_study.py --csv iag_history_5m.csv
```

Para años de una vez necesitarás un feed de pago o el export de tu bróker (intradía multi-año
de acciones no es gratis); una vez tengas el CSV, `--csv` lo procesa igual.

### Opciones principales

| Flag | Descripción |
|------|-------------|
| `--csv RUTA` | Usa un CSV propio (fecha-hora + OHLCV) y evita el límite de ~60 días de yfinance. |
| `--cost-bps 10` | Comisión ida+vuelta (bps). |
| `--spread-bps 3` | Spread bid/ask cruzado ida+vuelta (bps). |
| `--borrow-annual-pct 3` | Coste anual del préstamo del corto (prorrateado por tiempo). |
| `--stop-pct 1.0` | Stop-loss del corto en % en contra (0 = sin stop). |

El resumen incluye además del retorno neto: **Sharpe por operación, drawdown máximo,
profit factor**, contraste **evento vs no-evento** por permutación (`perm_pvalue`) y
recuento de salidas por stop.

> **Aviso:** 60 días de datos son solo exploratorios. Lee `VALORACION.md` antes de sacar conclusiones
> o arriesgar capital. Esto no es asesoramiento financiero.
