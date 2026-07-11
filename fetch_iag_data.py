#!/usr/bin/env python3
"""Acumulador incremental de velas intradía de IAG (o cualquier ticker).

yfinance solo ofrece ~60 días de velas de 5 minutos. Este script descarga esa
ventana rodante y la FUSIONA (sin duplicar) en un CSV creciente. Ejecutándolo
periódicamente (p. ej. una vez por semana) se va acumulando historia real que
luego puedes analizar con:

    python iag_intraday_event_study.py --csv iag_history_5m.csv

Uso:
    python fetch_iag_data.py                         # actualiza iag_history_5m.csv
    python fetch_iag_data.py --out datos.csv         # otro fichero
    python fetch_iag_data.py --ticker IAG.MC --interval 5m --period 60d

Requiere red hacia Yahoo Finance (no funciona en entornos con la salida
restringida). Instala dependencias con:  pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Reutiliza la descarga, la carga de CSV y la fusión del módulo principal.
from iag_intraday_event_study import Config, download_data, load_csv_data, merge_candles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Acumula velas intradía en un CSV creciente")
    p.add_argument("--ticker", default="IAG.MC")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d", help="Ventana a descargar (máx útil en 5m: 60d)")
    p.add_argument("--out", default="iag_history_5m.csv", help="CSV acumulado (se crea o se amplía)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(ticker=args.ticker, interval=args.interval, period=args.period)
    out = Path(args.out)

    try:
        fresh = download_data(cfg)
        print(f"Descargadas {len(fresh)} velas nuevas "
              f"({fresh.index.min()} -> {fresh.index.max()}).")

        if out.exists():
            existing = load_csv_data(cfg, str(out))
            before = len(existing)
            merged = merge_candles(existing, fresh)
            added = len(merged) - before
            print(f"CSV previo: {before} velas. Añadidas {added} nuevas.")
        else:
            merged = fresh.sort_index()
            print("No había CSV previo; se crea uno nuevo.")

        merged.to_csv(out, index_label="datetime")
        span_days = (merged.index.max() - merged.index.min()).days
        print(f"\nGuardado: {out.resolve()}")
        print(f"Total acumulado: {len(merged)} velas, {span_days} días de rango "
              f"({merged.index.min().date()} -> {merged.index.max().date()}).")
        print("Sugerencia: vuelve a ejecutar cada pocos días para ampliar la historia.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
