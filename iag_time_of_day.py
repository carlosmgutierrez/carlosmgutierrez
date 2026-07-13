#!/usr/bin/env python3
"""¿A qué hora del día está IAG de media más arriba y más abajo?

Estudio de estacionalidad intradía. Para cada instante de 5 minutos de la sesión
promedia, sobre todos los días, la cotización expresada como % respecto a la
apertura de ese día (así la deriva de largo plazo no distorsiona la "forma del día").

Uso:
    python iag_time_of_day.py --csv iag_5min_60d.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iag_intraday_event_study import Config, load_csv_data, parse_clock, split_complete_sessions


def hhmm(minutes: int) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def build_profile(csv_path: str, out_dir: Path) -> pd.DataFrame:
    cfg = Config()
    raw = load_csv_data(cfg, csv_path)
    sessions = split_complete_sessions(raw, cfg)
    open_ok = parse_clock("09:05")

    series_list = []
    for date in sorted(sessions):
        g = sessions[date].sort_index()
        if g.index[0].time() > open_ok:  # exige apertura real
            continue
        day_open = float(g["Open"].to_numpy()[0])
        norm = 100.0 * (g["Close"].to_numpy(dtype=float) / day_open - 1.0)
        minute = (g.index.hour * 60 + g.index.minute).to_numpy()
        s = pd.Series(norm, index=minute)
        s = s.groupby(level=0).last()  # por si hay duplicados
        series_list.append(s.rename(str(date.date())))

    if len(series_list) < 10:
        raise RuntimeError(f"Solo {len(series_list)} sesiones válidas; muy pocas.")

    mat = pd.concat(series_list, axis=1).sort_index()
    n_days = mat.shape[1]
    prof = pd.DataFrame({
        "minute": mat.index,
        "clock": [hhmm(m) for m in mat.index],
        "mean_pct": mat.mean(axis=1).to_numpy(),
        "median_pct": mat.median(axis=1).to_numpy(),
        "std_pct": mat.std(axis=1, ddof=1).to_numpy(),
        "n_days": mat.count(axis=1).to_numpy(),
    })
    # Solo instantes presentes en al menos el 60% de los días (evita bordes ruidosos).
    prof = prof[prof["n_days"] >= max(10, int(0.6 * n_days))].reset_index(drop=True)
    prof["stderr_pct"] = prof["std_pct"] / np.sqrt(prof["n_days"])

    out_dir.mkdir(parents=True, exist_ok=True)
    prof.to_csv(out_dir / "time_of_day_profile.csv", index=False)
    return prof, n_days


def make_figure(prof: pd.DataFrame, n_days: int, out_dir: Path) -> tuple[pd.Series, pd.Series]:
    hi = prof.loc[prof["mean_pct"].idxmax()]
    lo = prof.loc[prof["mean_pct"].idxmin()]

    x = prof["minute"].to_numpy()
    y = prof["mean_pct"].to_numpy()
    se = prof["stderr_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, y - se, y + se, alpha=0.2, label="±1 error estándar")
    ax.plot(x, y, linewidth=2, label=f"Media de {n_days} días")
    ax.axhline(0, color="gray", linewidth=1)
    ax.scatter([hi["minute"]], [hi["mean_pct"]], color="green", zorder=5, s=70)
    ax.scatter([lo["minute"]], [lo["mean_pct"]], color="red", zorder=5, s=70)
    ax.annotate(f"MÁS ALTA\n{hi['clock']}  ({hi['mean_pct']:+.2f}%)",
                (hi["minute"], hi["mean_pct"]), textcoords="offset points",
                xytext=(0, 14), ha="center", color="green", fontweight="bold")
    ax.annotate(f"MÁS BAJA\n{lo['clock']}  ({lo['mean_pct']:+.2f}%)",
                (lo["minute"], lo["mean_pct"]), textcoords="offset points",
                xytext=(0, -28), ha="center", color="red", fontweight="bold")

    ticks = np.arange((x.min() // 30) * 30, x.max() + 1, 30)
    ax.set_xticks(ticks)
    ax.set_xticklabels([hhmm(t) for t in ticks], rotation=45)
    ax.set_xlabel("Hora (Madrid)")
    ax.set_ylabel("Cotización media respecto a la apertura (%)")
    ax.set_title("Forma media del día en IAG: ¿a qué hora está de media más arriba/abajo?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "time_of_day_profile.png", dpi=160)
    plt.close(fig)
    return hi, lo


def main() -> int:
    p = argparse.ArgumentParser(description="Estacionalidad intradía de IAG (hora más alta/baja)")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="iag_time_of_day_results")
    args = p.parse_args()
    try:
        prof, n_days = build_profile(args.csv, Path(args.output))
        hi, lo = make_figure(prof, n_days, Path(args.output))
        print(f"Días analizados: {n_days}")
        print(f"HORA MÁS ALTA de media: {hi['clock']}  ->  {hi['mean_pct']:+.3f}% sobre la apertura")
        print(f"HORA MÁS BAJA de media: {lo['clock']}  ->  {lo['mean_pct']:+.3f}% sobre la apertura")
        print(f"\nApertura (09:00) = 0% por definición. Cierre medio: "
              f"{prof['mean_pct'].iloc[-1]:+.3f}% ({prof['clock'].iloc[-1]}).")
        print(f"Resultados en: {Path(args.output).resolve()}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
