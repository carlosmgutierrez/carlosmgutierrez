#!/usr/bin/env python3
"""Verificación de la LÓGICA de iag_intraday_event_study.py con datos sintéticos.

No requiere red (yfinance no se usa: se reemplaza la descarga por un generador).
Sirve para comprobar en cualquier entorno que:

  1) el pipeline corre de extremo a extremo;
  2) NO hay look-ahead (ninguna salida ocurre antes o a la vez que la entrada);
  3) el SIGNO del corto es correcto: si se inyecta un patrón "sube por la mañana
     y revierte por la tarde", la estrategia corta debe salir GANADORA.

Uso:
    python tests/synthetic_check.py

Devuelve código de salida 0 si todas las comprobaciones pasan.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "iag_intraday_event_study.py"

spec = importlib.util.spec_from_file_location("study", MOD)
study = importlib.util.module_from_spec(spec)
sys.modules["study"] = study
spec.loader.exec_module(study)

rng = np.random.default_rng(42)


def make_session(day, inject_reversion=True):
    """Genera velas 5m de 09:00 a 17:25 para una sesión."""
    times = pd.date_range(f"{day} 09:00", f"{day} 17:25", freq="5min", tz="Europe/Madrid")
    n = len(times)
    open0 = 100.0
    strong_up = rng.random() < 0.40
    drift_am = (0.015 if strong_up else 0.0) + rng.normal(0, 0.001)
    drift_pm = (-0.020 if (strong_up and inject_reversion) else rng.normal(0, 0.0008))
    prices = [open0]
    for i in range(1, n):
        t = times[i]
        mins = t.hour * 60 + t.minute
        mu = drift_am / 6 if mins <= 9 * 60 + 30 else drift_pm / (n - 6)
        prices.append(prices[-1] * (1 + mu + rng.normal(0, 0.0003)))
    close = np.array(prices)
    op = np.concatenate([[open0], close[:-1]])
    high = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.0004, n)))
    low = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.0004, n)))
    vol = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol}, index=times
    )


def main() -> int:
    days = pd.bdate_range("2026-05-01", periods=45)
    raw = pd.concat([make_session(d.date()) for d in days]).sort_index()
    study.yf.download = lambda *a, **k: raw.copy()  # sin red

    cfg = study.Config()
    df = study.download_data(cfg)
    sessions = study.split_complete_sessions(df, cfg)
    trades = study.build_trade_table(sessions, cfg)
    summary = study.summarize_strategy(trades, cfg)

    ok = True

    # (1) end-to-end
    assert len(sessions) == 45, f"esperadas 45 sesiones, hay {len(sessions)}"
    print(f"[OK] pipeline completo: {len(sessions)} sesiones, {len(trades)} observaciones")

    # (2) sin look-ahead
    n_bad = int((pd.to_datetime(trades["exit_time"]) <= pd.to_datetime(trades["entry_time"])).sum())
    if n_bad != 0:
        ok = False
        print(f"[FALLO] {n_bad} filas con salida <= entrada (look-ahead)")
    else:
        print("[OK] sin look-ahead: toda salida es posterior a la entrada")

    # (3) signo del corto correcto
    main_cfg = summary[
        (summary.checkpoint == "09:30")
        & (summary.threshold_pct == 1.0)
        & (summary.exit == "17:25")
    ]
    mean_net = float(main_cfg["mean_net_short_pct"].iloc[0])
    edge = float(main_cfg["edge_vs_all_days_pct"].iloc[0])
    if mean_net > 0 and edge > 0:
        print(f"[OK] signo correcto: corto medio {mean_net:.2f}% (edge {edge:+.2f}%) en reversión inyectada")
    else:
        ok = False
        print(f"[FALLO] con reversión inyectada el corto debería ganar; media {mean_net:.2f}%")

    # (4) columnas y test evento vs no-evento
    nuevas = {"sharpe_per_trade", "event_minus_nonevent_pct", "perm_pvalue",
              "max_drawdown_pct", "profit_factor", "n_stop_exits"}
    faltan = nuevas - set(summary.columns)
    if faltan:
        ok = False
        print(f"[FALLO] faltan columnas nuevas: {sorted(faltan)}")
    else:
        pval = float(main_cfg["perm_pvalue"].iloc[0])
        diff = float(main_cfg["event_minus_nonevent_pct"].iloc[0])
        if pval < 0.05 and diff > 0:
            print(f"[OK] evento vs no-evento significativo: dif {diff:+.2f}%, p={pval:.4f}")
        else:
            ok = False
            print(f"[FALLO] esperado evento>no-evento significativo; dif {diff:+.2f}%, p={pval:.4f}")

    # (5) el stop-loss dispara salidas anticipadas
    cfg_stop = study.Config(stop_loss_pct=0.5)
    trades_stop = study.build_trade_table(sessions, cfg_stop)
    n_stops = int((trades_stop["exit_reason"] == "stop").sum())
    if n_stops > 0 and (trades_stop["exit_reason"] == "time").any():
        print(f"[OK] stop-loss operativo: {n_stops} salidas por stop de {len(trades_stop)} operaciones")
    else:
        ok = False
        print(f"[FALLO] el stop-loss no generó salidas anticipadas (n_stops={n_stops})")

    # (6) acumulador: fusión sin duplicados + ciclo CSV -> recarga -> fusión
    import tempfile
    cfg0 = study.Config()
    ventana1 = study.download_data(cfg0)  # yf está parcheado: usa 'raw'
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "hist.csv"
        ventana1.to_csv(p, index_label="datetime")
        recargada = study.load_csv_data(cfg0, str(p))
        # segunda ventana: mismos datos + un día extra nuevo (posterior al rango)
        next_day = recargada.index.max().normalize() + pd.Timedelta(days=5)
        while next_day.weekday() >= 5:  # asegura día laborable
            next_day += pd.Timedelta(days=1)
        extra = make_session(next_day.date())
        ventana2 = study.normalize_frame(
            pd.concat([recargada, extra]).sort_index(), cfg0, assume_tz="Europe/Madrid"
        )
        fusion = study.merge_candles(recargada, ventana2)
        sin_dup = not fusion.index.duplicated().any()
        crece = len(fusion) > len(recargada)
        if sin_dup and crece:
            print(f"[OK] acumulador: {len(recargada)} -> {len(fusion)} velas, sin duplicados")
        else:
            ok = False
            print(f"[FALLO] acumulador: dup={not sin_dup}, crece={crece}")

    print("\nRESULTADO:", "TODAS LAS COMPROBACIONES PASAN" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
