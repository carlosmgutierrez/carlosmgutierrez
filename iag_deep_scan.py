#!/usr/bin/env python3
"""Exploración profunda de estrategias intradía CORTAS en IAG (vender y recomprar
el mismo día, siempre plano antes del cierre).

Explora muchas combinaciones de reglas de ENTRADA (abrir corto) y SALIDA (cubrir):

  ENTRADA
    - Por hora fija: vender a una hora del reloj (todos los días).
    - Por subida fija: vender solo si el precio ha subido >= X% desde la apertura
      a una hora de observación dada (la señal usa solo velas cerradas; se entra
      en la apertura de la vela siguiente -> sin look-ahead).

  SALIDA (siempre antes del cierre; tope duro a las 17:25)
    - Por hora fija.
    - Por objetivo de beneficio del corto (take-profit): cubrir si el precio baja Y%.
    - Opcionalmente con stop: cubrir si el precio sube Z% en contra.

Para cada combinación calcula retorno neto (con costes), win rate, Sharpe, drawdown,
profit factor y frecuencia. Y hace la prueba anti-sobreajuste clave: ordena por la
PRIMERA mitad de los días (in-sample) y mide la SEGUNDA mitad (out-of-sample).

Uso:
    python iag_deep_scan.py --csv iag_5min_60d.csv
    python iag_deep_scan.py --csv datos.csv --min-trades 8 --output scan_out
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iag_intraday_event_study import (
    Config,
    equity_curve_metrics,
    interval_minutes,
    load_csv_data,
    minutes_from_midnight,
    parse_clock,
    split_complete_sessions,
)

# --------------------------- Rejillas de exploración ---------------------------
ENTRY_TIMES = ("09:15", "09:30", "09:45", "10:00", "10:30", "11:00", "12:00")
RISE_THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)          # "subidas fijas" (%)
FIXED_EXIT_TIMES = ("11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:25")
TAKE_PROFITS = (0.25, 0.5, 0.75, 1.0, 1.5)                   # cubrir si el corto gana Y%
STOPS = (0.0, 1.0, 1.5)                                       # 0 = sin stop
EOD_CAP = "17:25"


# --------------------------- Preparación de sesiones ---------------------------
def session_arrays(group: pd.DataFrame, bar_minutes: int) -> dict:
    starts = (group.index.hour * 60 + group.index.minute).to_numpy()
    return {
        "starts": starts,
        "ends": starts + bar_minutes,
        "open": group["Open"].to_numpy(dtype=float),
        "high": group["High"].to_numpy(dtype=float),
        "low": group["Low"].to_numpy(dtype=float),
        "close": group["Close"].to_numpy(dtype=float),
        "day_open": float(group["Open"].to_numpy(dtype=float)[0]),
    }


def find_entry(sa: dict, entry_kind: str, entry_time_min: int, rise_thr: float):
    """Devuelve (idx, entry_price, entry_min) o None si no hay señal ese día."""
    if entry_kind == "rise":
        closed = sa["ends"] <= entry_time_min
        if not closed.any():
            return None
        signal_price = sa["close"][closed][-1]
        rise = 100.0 * (signal_price / sa["day_open"] - 1.0)
        if rise < rise_thr:
            return None
    # entrada en la apertura de la primera vela que empieza en/después de la hora
    cand = np.nonzero(sa["starts"] >= entry_time_min)[0]
    if len(cand) == 0:
        return None
    i = cand[0]
    return i, float(sa["open"][i]), int(sa["starts"][i])


def find_exit(sa: dict, entry_idx: int, entry_price: float, entry_min: int,
              exit_kind: str, exit_param, tp: float, stop: float, eod_min: int):
    """Devuelve (exit_price, exit_min) o None. Corto: gana si el precio baja."""
    if exit_kind == "time":
        h_out = exit_param
        closed = (sa["ends"] <= h_out) & (sa["ends"] > entry_min)
        if not closed.any():
            return None
        j = np.nonzero(closed)[0][-1]
        return float(sa["close"][j]), int(sa["ends"][j])

    # exit_kind == "tp": recorre velas desde la entrada; TP y/o stop; si no, EOD.
    tp_price = entry_price * (1.0 - tp / 100.0)
    stop_price = entry_price * (1.0 + stop / 100.0) if stop > 0 else None
    n = len(sa["starts"])
    for j in range(entry_idx, n):
        if sa["ends"][j] > eod_min:
            break
        if sa["ends"][j] <= entry_min:
            continue
        # Si en la misma vela tocan stop y TP, asumimos stop primero (conservador).
        if stop_price is not None and sa["high"][j] >= stop_price:
            return stop_price, int(sa["ends"][j])
        if sa["low"][j] <= tp_price:
            return tp_price, int(sa["ends"][j])
    # Cierre forzado al final de la sesión (última vela cerrada <= EOD).
    closed = (sa["ends"] <= eod_min) & (sa["ends"] > entry_min)
    if not closed.any():
        return None
    j = np.nonzero(closed)[0][-1]
    return float(sa["close"][j]), int(sa["ends"][j])


def simulate_combo(sessions_arrays, combo, cost, eod_min):
    """Retornos netos por sesión (%). combo: dict con la definición de la estrategia."""
    rets, entry_ok = [], 0
    for sa in sessions_arrays:
        ent = find_entry(sa, combo["entry_kind"], combo["entry_min"], combo["rise_thr"])
        if ent is None:
            continue
        entry_idx, entry_price, entry_min = ent
        ex = find_exit(sa, entry_idx, entry_price, entry_min,
                       combo["exit_kind"], combo["exit_param"], combo["tp"],
                       combo["stop"], eod_min)
        if ex is None:
            continue
        exit_price, exit_min = ex
        entry_ok += 1
        gross = entry_price / exit_price - 1.0
        borrow = (cost["borrow_annual_pct"] / 100.0) * ((exit_min - entry_min) / 525_600.0)
        net = gross - cost["commission"] - cost["spread"] - borrow
        rets.append(100.0 * net)
    return np.array(rets, dtype=float)


def stats_from_returns(rets: np.ndarray) -> dict:
    n = len(rets)
    if n == 0:
        return {"n_trades": 0}
    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if n > 1 else math.nan
    max_dd, pf = equity_curve_metrics(rets)
    return {
        "n_trades": n,
        "mean_net_pct": mean,
        "median_net_pct": float(np.median(rets)),
        "win_rate_pct": 100.0 * float(np.mean(rets > 0)),
        "sharpe_per_trade": (mean / std) if (n > 1 and std > 0) else math.nan,
        "std_pct": std,
        "max_drawdown_pct": max_dd,
        "profit_factor": pf,
        "compounded_pct": 100.0 * (np.prod(1.0 + rets / 100.0) - 1.0),
    }


def enumerate_combos() -> list[dict]:
    combos: list[dict] = []
    # Familia A: hora fija -> hora fija (vender a H_in, cubrir a H_out).
    for h_in in ENTRY_TIMES:
        for h_out in FIXED_EXIT_TIMES:
            if minutes_from_midnight(h_out) <= minutes_from_midnight(h_in):
                continue
            combos.append({
                "family": "hora->hora", "entry_kind": "time", "entry_time": h_in,
                "entry_min": minutes_from_midnight(h_in), "rise_thr": 0.0,
                "exit_kind": "time", "exit_param": minutes_from_midnight(h_out),
                "exit_label": h_out, "tp": 0.0, "stop": 0.0,
            })
    # Familia B: subida fija -> hora fija.
    for h_in in ENTRY_TIMES:
        for thr in RISE_THRESHOLDS:
            for h_out in FIXED_EXIT_TIMES:
                if minutes_from_midnight(h_out) <= minutes_from_midnight(h_in):
                    continue
                combos.append({
                    "family": "subida->hora", "entry_kind": "rise", "entry_time": h_in,
                    "entry_min": minutes_from_midnight(h_in), "rise_thr": thr,
                    "exit_kind": "time", "exit_param": minutes_from_midnight(h_out),
                    "exit_label": h_out, "tp": 0.0, "stop": 0.0,
                })
    # Familia C: subida fija -> objetivo de beneficio (con/sin stop), tope EOD.
    for h_in in ENTRY_TIMES:
        for thr in RISE_THRESHOLDS:
            for tp in TAKE_PROFITS:
                for stop in STOPS:
                    combos.append({
                        "family": "subida->objetivo", "entry_kind": "rise",
                        "entry_time": h_in, "entry_min": minutes_from_midnight(h_in),
                        "rise_thr": thr, "exit_kind": "tp", "exit_param": None,
                        "exit_label": f"TP{tp}%/SL{stop}%", "tp": tp, "stop": stop,
                    })
    return combos


def make_figures(res: pd.DataFrame, out_dir: Path) -> None:
    # 1) Mapa de calor familia hora->hora: retorno medio por (entrada, salida).
    fam = res[res["family"] == "hora->hora"]
    if not fam.empty:
        pivot = fam.pivot_table(index="entry_time", columns="exit",
                                values="mean_net_pct", aggfunc="mean")
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        fig, ax = plt.subplots(figsize=(9, 5))
        vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Hora de salida (cubrir)")
        ax.set_ylabel("Hora de entrada (vender)")
        ax.set_title("Retorno neto medio del corto (%) — vender a hora fija, cubrir a hora fija\n"
                     "Azul = pierde, Rojo = gana")
        fig.colorbar(im, ax=ax, label="Retorno neto medio (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "heatmap_hora_hora.png", dpi=160)
        plt.close(fig)

    # 2) In-sample vs out-of-sample: si no hay relación, es sobreajuste.
    v = res.dropna(subset=["is_mean_pct", "oos_mean_pct"])
    if not v.empty:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(v["is_mean_pct"], v["oos_mean_pct"], s=14, alpha=0.5)
        ax.axhline(0, linewidth=1, color="gray")
        ax.axvline(0, linewidth=1, color="gray")
        lim = float(np.nanmax(np.abs(np.r_[v["is_mean_pct"], v["oos_mean_pct"]]))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "--", color="black", linewidth=1, label="y = x (ideal)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Retorno medio 1ª mitad (in-sample) %")
        ax.set_ylabel("Retorno medio 2ª mitad (out-of-sample) %")
        corr = v["is_mean_pct"].corr(v["oos_mean_pct"])
        ax.set_title(f"¿Lo que gana en la 1ª mitad gana en la 2ª?\ncorrelación = {corr:+.2f} "
                     f"(≈0 => sobreajuste, no hay señal real)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "in_sample_vs_out_of_sample.png", dpi=160)
        plt.close(fig)


def run(csv_path: str, out_dir: Path, min_trades: int) -> int:
    cfg = Config()
    raw = load_csv_data(cfg, csv_path)
    sessions = split_complete_sessions(raw, cfg)
    bar_minutes = interval_minutes(cfg.interval)
    eod_min = minutes_from_midnight(EOD_CAP)
    open_start = parse_clock("09:05")

    # Solo sesiones con apertura real disponible (evita el día parcial del borde).
    dates = sorted(sessions)
    prepared = []
    for d in dates:
        g = sessions[d].sort_index()
        if g.index[0].time() > open_start:
            continue
        prepared.append((d, session_arrays(g, bar_minutes)))
    if len(prepared) < 12:
        raise RuntimeError(f"Solo {len(prepared)} sesiones válidas; muy pocas.")

    all_arrays = [sa for _, sa in prepared]
    half = len(prepared) // 2
    is_arrays = [sa for _, sa in prepared[:half]]        # in-sample (1a mitad)
    oos_arrays = [sa for _, sa in prepared[half:]]       # out-of-sample (2a mitad)

    cost = {"commission": cfg.round_trip_cost_bps / 10_000.0,
            "spread": cfg.spread_bps / 10_000.0,
            "borrow_annual_pct": cfg.borrow_annual_pct}

    combos = enumerate_combos()
    print(f"Sesiones válidas: {len(prepared)} (IS={len(is_arrays)}, OOS={len(oos_arrays)})")
    print(f"Combinaciones a explorar: {len(combos)}")

    rows = []
    for c in combos:
        full = stats_from_returns(simulate_combo(all_arrays, c, cost, eod_min))
        if full.get("n_trades", 0) < min_trades:
            continue
        is_m = stats_from_returns(simulate_combo(is_arrays, c, cost, eod_min))
        oos_m = stats_from_returns(simulate_combo(oos_arrays, c, cost, eod_min))
        rows.append({
            "family": c["family"], "entry_time": c["entry_time"],
            "rise_threshold_pct": c["rise_thr"], "exit": c["exit_label"],
            **full,
            "is_mean_pct": is_m.get("mean_net_pct", math.nan),
            "is_n": is_m.get("n_trades", 0),
            "oos_mean_pct": oos_m.get("mean_net_pct", math.nan),
            "oos_n": oos_m.get("n_trades", 0),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("Ninguna combinación alcanzó el mínimo de operaciones.")
    res = res.sort_values("mean_net_pct", ascending=False).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "deep_scan_results.csv", index=False)
    make_figures(res, out_dir)

    # ---- Informe por pantalla ----
    show = ["family", "entry_time", "rise_threshold_pct", "exit", "n_trades",
            "mean_net_pct", "win_rate_pct", "sharpe_per_trade", "max_drawdown_pct"]
    print(f"\n=== TOP 12 combinaciones (retorno neto medio del corto, muestra completa) ===")
    print(res[show].head(12).to_string(index=False))

    n_pos = int((res["mean_net_pct"] > 0).sum())
    print(f"\nDe {len(res)} combinaciones con >= {min_trades} operaciones, "
          f"{n_pos} ({100*n_pos/len(res):.0f}%) tienen retorno medio POSITIVO.")

    # ---- Prueba anti-sobreajuste: ordenar por IS, medir OOS ----
    valid = res[(res["is_n"] >= max(4, min_trades // 2)) & (res["oos_n"] >= max(4, min_trades // 2))]
    if not valid.empty:
        by_is = valid.sort_values("is_mean_pct", ascending=False)
        topk = by_is.head(10)
        print("\n=== PRUEBA OUT-OF-SAMPLE: top 10 por la 1a mitad, ¿aguantan en la 2a? ===")
        oos_cols = ["family", "entry_time", "rise_threshold_pct", "exit",
                    "is_mean_pct", "is_n", "oos_mean_pct", "oos_n"]
        print(topk[oos_cols].to_string(index=False))
        oos_pos = int((topk["oos_mean_pct"] > 0).sum())
        print(f"\nDe las 10 mejores in-sample, {oos_pos}/10 siguen POSITIVAS out-of-sample.")
        print(f"Correlación IS vs OOS (retorno medio): "
              f"{valid['is_mean_pct'].corr(valid['oos_mean_pct']):+.2f} "
              f"(cercano a 0 o negativo => el ranking no se sostiene => sobreajuste).")

    print(f"\nResultados completos: {(out_dir / 'deep_scan_results.csv').resolve()}")
    print("AVISO: explorar cientos de combos en 60 días infla el azar. La prueba "
          "out-of-sample de arriba es la que manda, no el 'top' de la muestra completa.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exploración profunda de estrategias intradía cortas")
    p.add_argument("--csv", required=True, help="CSV de velas (formato yfinance o plano)")
    p.add_argument("--min-trades", type=int, default=8, help="Mínimo de operaciones por combo")
    p.add_argument("--output", default="iag_deep_scan_results")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args.csv, Path(args.output), args.min_trades)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
