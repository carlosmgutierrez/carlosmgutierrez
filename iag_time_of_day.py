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
    meta_rows = []
    prev_close = None  # cierre de la sesión anterior, para el gap de apertura
    for date in sorted(sessions):
        g = sessions[date].sort_index()
        day_open = float(g["Open"].to_numpy()[0])
        day_close = float(g["Close"].to_numpy()[-1])
        gap = None if prev_close is None else 100.0 * (day_open / prev_close - 1.0)
        if g.index[0].time() <= open_ok:  # exige apertura real para la trayectoria
            norm = 100.0 * (g["Close"].to_numpy(dtype=float) / day_open - 1.0)
            minute = (g.index.hour * 60 + g.index.minute).to_numpy()
            s = pd.Series(norm, index=minute)
            s = s.groupby(level=0).last()  # por si hay duplicados
            series_list.append(s.rename(str(date.date())))
            meta_rows.append({"date": str(date.date()), "gap_pct": gap})
        prev_close = day_close  # se actualiza siempre (también en días descartados)

    if len(series_list) < 10:
        raise RuntimeError(f"Solo {len(series_list)} sesiones válidas; muy pocas.")

    mat = pd.concat(series_list, axis=1).sort_index()
    meta = pd.DataFrame(meta_rows).set_index("date")
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
    return prof, n_days, mat, meta


def _group_mean(mat: pd.DataFrame, cols: list[str], min_frac: float = 0.6) -> pd.DataFrame:
    sub = mat[cols]
    n = len(cols)
    out = pd.DataFrame({
        "minute": sub.index,
        "mean_pct": sub.mean(axis=1).to_numpy(),
        "n": sub.count(axis=1).to_numpy(),
        "stderr_pct": (sub.std(axis=1, ddof=1) / np.sqrt(sub.count(axis=1))).to_numpy(),
    })
    return out[out["n"] >= max(5, int(min_frac * n))].reset_index(drop=True)


def _plot_two_groups(mat, pos_cols, neg_cols, pos_label, neg_label,
                     title, filename, out_dir):
    pos = _group_mean(mat, pos_cols) if pos_cols else pd.DataFrame()
    neg = _group_mean(mat, neg_cols) if neg_cols else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(11, 6))
    for grp, color, label in ((pos, "green", pos_label), (neg, "red", neg_label)):
        if grp.empty:
            continue
        x, y, se = grp["minute"].to_numpy(), grp["mean_pct"].to_numpy(), grp["stderr_pct"].to_numpy()
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.12)

    ax.axhline(0, color="gray", linewidth=1)
    all_min = mat.index.to_numpy()
    ticks = np.arange((all_min.min() // 30) * 30, all_min.max() + 1, 30)
    ax.set_xticks(ticks)
    ax.set_xticklabels([hhmm(t) for t in ticks], rotation=45)
    ax.set_xlabel("Hora (Madrid)")
    ax.set_ylabel("Cotización media respecto a la apertura (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)

    summary = {}
    if not pos.empty:
        summary["pos_close"] = (hhmm(pos["minute"].iloc[-1]), float(pos["mean_pct"].iloc[-1]))
    if not neg.empty:
        summary["neg_close"] = (hhmm(neg["minute"].iloc[-1]), float(neg["mean_pct"].iloc[-1]))
    return summary


def make_gap_figure(mat: pd.DataFrame, meta: pd.DataFrame, out_dir: Path):
    """Separa días por el signo del GAP de apertura (open vs cierre anterior).

    El gap se conoce a las 09:00, sin usar información futura.
    """
    gap = meta["gap_pct"]
    pos_cols = [c for c in mat.columns if c in gap.index and pd.notna(gap[c]) and gap[c] > 0]
    neg_cols = [c for c in mat.columns if c in gap.index and pd.notna(gap[c]) and gap[c] < 0]
    summary = _plot_two_groups(
        mat, pos_cols, neg_cols,
        pos_label=f"Abre en positivo / gap+ (N={len(pos_cols)})",
        neg_label=f"Abre en negativo / gap− (N={len(neg_cols)})",
        title="Forma media del día en IAG según cómo ABRE (gap de apertura)\n"
              "(clasificación sin información futura)",
        filename="time_of_day_by_open_gap.png", out_dir=out_dir,
    )
    summary["pos_days"] = len(pos_cols)
    summary["neg_days"] = len(neg_cols)
    return summary


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
        prof, n_days, mat, meta = build_profile(args.csv, Path(args.output))
        hi, lo = make_figure(prof, n_days, Path(args.output))
        print(f"Días analizados: {n_days}")
        print(f"HORA MÁS ALTA de media: {hi['clock']}  ->  {hi['mean_pct']:+.3f}% sobre la apertura")
        print(f"HORA MÁS BAJA de media: {lo['clock']}  ->  {lo['mean_pct']:+.3f}% sobre la apertura")
        print(f"\nApertura (09:00) = 0% por definición. Cierre medio: "
              f"{prof['mean_pct'].iloc[-1]:+.3f}% ({prof['clock'].iloc[-1]}).")

        d = make_gap_figure(mat, meta, Path(args.output))
        print(f"\n--- Separando por el GAP de apertura (sin info futura) ---")
        print(f"Abre en positivo: {d['pos_days']}   |   Abre en negativo: {d['neg_days']}")
        if "pos_close" in d:
            print(f"Abre en positivo: cierre medio {d['pos_close'][1]:+.3f}% (a las {d['pos_close'][0]})")
        if "neg_close" in d:
            print(f"Abre en negativo: cierre medio {d['neg_close'][1]:+.3f}% (a las {d['neg_close'][0]})")
        print(f"\nResultados en: {Path(args.output).resolve()}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
