# Estudio intradía de IAG (BME)

Backtest exploratorio de un patrón de **reversión intradía** en IAG (`IAG.MC`): tras una subida
matinal, ¿tiende el valor a caer hasta el cierre? Posición **corta**, velas de 5 minutos.

## Contenido

| Fichero | Descripción |
|---------|-------------|
| `iag_intraday_event_study.py` | Estudio de eventos: descarga, señal sin look-ahead, entrada/salida, estadística y figuras. |
| `INSTRUCCIONES_IAG_INTRADIA.txt` | Instrucciones de uso originales. |
| `VALORACION.md` | **Informe de valoración** de la estrategia (fortalezas, debilidades y recomendaciones). |
| `tests/synthetic_check.py` | Verificación de la lógica con datos sintéticos (no necesita red). |
| `requirements.txt` | Dependencias. |

## Uso rápido

```bash
pip install -r requirements.txt
python iag_intraday_event_study.py     # necesita acceso de red a Yahoo Finance
python tests/synthetic_check.py        # comprueba la lógica sin red
```

Los resultados se guardan en `iag_intraday_results/` (CSV + PNG).

> **Aviso:** 60 días de datos son solo exploratorios. Lee `VALORACION.md` antes de sacar conclusiones
> o arriesgar capital. Esto no es asesoramiento financiero.
