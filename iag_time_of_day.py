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

from iag_intraday_event_study import (
    Config,
    load_csv_data,
    minutes_from_midnight,
    parse_clock,
    split_complete_sessions,
)


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
                     title, filename, out_dir,
                     ylabel="Cotización media respecto a la apertura (%)"):
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
    ax.set_ylabel(ylabel)
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


def make_gap_vs_prevclose_figure(mat: pd.DataFrame, meta: pd.DataFrame, out_dir: Path):
    """Igual que la anterior pero normalizando al CIERRE ANTERIOR, así el gap se ve:
    los días que abren en negativo empiezan por DEBAJO de 0."""
    gap = meta["gap_pct"]
    cols = [c for c in mat.columns if c in gap.index and pd.notna(gap[c])]
    # path_vs_prevclose = (1 + path_vs_open) * (1 + gap) - 1
    matp = pd.DataFrame(index=mat.index)
    for c in cols:
        matp[c] = 100.0 * ((1.0 + mat[c] / 100.0) * (1.0 + gap[c] / 100.0) - 1.0)
    pos_cols = [c for c in cols if gap[c] > 0]
    neg_cols = [c for c in cols if gap[c] < 0]
    return _plot_two_groups(
        matp, pos_cols, neg_cols,
        pos_label=f"Abre en positivo / gap+ (N={len(pos_cols)})",
        neg_label=f"Abre en negativo / gap− (N={len(neg_cols)})",
        title="Forma del día en IAG respecto al CIERRE ANTERIOR (el gap SÍ se ve)\n"
              "Los días que abren en negativo empiezan por debajo de 0",
        filename="time_of_day_by_open_gap_vs_prevclose.png", out_dir=out_dir,
        ylabel="Cotización media respecto al cierre anterior (%)",
    )


def make_spaghetti_figures(mat: pd.DataFrame, out_dir: Path) -> dict:
    """Dos figuras con TODAS las trayectorias individuales superpuestas:
    una para días verdes (cierre > apertura) y otra para rojos (cierre < apertura)."""
    up_cols, down_cols = [], []
    for c in mat.columns:
        s = mat[c].dropna()
        if s.empty:
            continue
        (up_cols if s.iloc[-1] > 0 else down_cols).append(c)

    all_min = mat.index.to_numpy()
    ticks = np.arange((all_min.min() // 30) * 30, all_min.max() + 1, 30)

    for cols, color, nombre, fname in (
        (up_cols, "green", "verdes (cierre > apertura)", "spaghetti_dias_verdes.png"),
        (down_cols, "red", "rojos (cierre < apertura)", "spaghetti_dias_rojos.png"),
    ):
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        for c in cols:
            ax.plot(mat.index.to_numpy(), mat[c].to_numpy(), color=color,
                    alpha=0.28, linewidth=0.8)
        ax.plot(mat.index.to_numpy(), mat[cols].mean(axis=1).to_numpy(),
                color="black", linewidth=2.5, label="Media")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([hhmm(t) for t in ticks], rotation=45)
        ax.set_xlabel("Hora (Madrid)")
        ax.set_ylabel("Cotización respecto a la apertura (%)")
        ax.set_title(f"Todas las trayectorias de días {nombre}  (N={len(cols)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)

    return {"verdes": len(up_cols), "rojos": len(down_cols)}


def make_weekly_grid(mat: pd.DataFrame, out_dir: Path) -> int:
    """Rejilla de paneles, uno por semana natural, con las trayectorias de esa
    semana en verde (cierre>apertura) o rojo (cierre<apertura). Ejes compartidos
    para comparar de un vistazo. Cada línea se etiqueta con su día (L M X J V)."""
    import math as _math

    weeks: dict[tuple, list] = {}
    for c in mat.columns:
        ts = pd.Timestamp(c)
        iso = ts.isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append((ts, c))
    order = sorted(weeks)
    nw = len(order)
    if nw == 0:
        return 0

    ncols = 3
    nrows = _math.ceil(nw / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.1 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    x = mat.index.to_numpy()
    ticks = [minutes_from_midnight(t) for t in ("09:00", "12:00", "15:00", "17:00")]
    wd = {0: "L", 1: "M", 2: "X", 3: "J", 4: "V", 5: "S", 6: "D"}

    for i, wk in enumerate(order):
        ax = axes[i]
        for ts, c in sorted(weeks[wk]):
            s = mat[c].dropna()
            if s.empty:
                continue
            color = "green" if s.iloc[-1] > 0 else "red"
            ax.plot(s.index.to_numpy(), s.to_numpy(), color=color, linewidth=1.4, alpha=0.9)
            ax.annotate(wd[ts.weekday()], (s.index[-1], s.iloc[-1]), fontsize=8,
                        color=color, xytext=(3, 0), textcoords="offset points", va="center")
        ax.axhline(0, color="gray", linewidth=0.8)
        first_day = min(t for t, _ in weeks[wk])
        ax.set_title(f"Sem {i + 1} · desde {first_day.strftime('%d-%b')}", fontsize=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([hhmm(t) for t in ticks], fontsize=8)
        ax.grid(True, alpha=0.25)

    for j in range(nw, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Trayectorias intradía de IAG por semana  (verde: cierra arriba · rojo: cierra abajo)",
                 fontsize=13)
    fig.supxlabel("Hora (Madrid)")
    fig.supylabel("Cotización respecto a la apertura (%)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "trayectorias_por_semana.png", dpi=150)
    plt.close(fig)
    return nw


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

        d2 = make_gap_vs_prevclose_figure(mat, meta, Path(args.output))
        print(f"\n--- Respecto al cierre anterior (el gap se ve) ---")
        if "pos_close" in d2:
            print(f"Abre en positivo: acaba el día a {d2['pos_close'][1]:+.3f}% del cierre anterior")
        if "neg_close" in d2:
            print(f"Abre en negativo: acaba el día a {d2['neg_close'][1]:+.3f}% del cierre anterior")
        sp = make_spaghetti_figures(mat, Path(args.output))
        print(f"\n--- Trayectorias superpuestas (espagueti) ---")
        print(f"Figuras generadas: {sp['verdes']} días verdes y {sp['rojos']} días rojos.")

        nw = make_weekly_grid(mat, Path(args.output))
        print(f"\n--- Rejilla por semanas ---")
        print(f"Generado trayectorias_por_semana.png con {nw} semanas.")
        print(f"\nResultados en: {Path(args.output).resolve()}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
