#!/usr/bin/env python3
"""
Estudio de eventos intradía para IAG en BME (ticker Yahoo: IAG.MC).

Hipótesis principal:
    Si IAG sube X% desde la apertura hasta una hora dada, ¿tiende a caer
    posteriormente hasta una hora de salida fija?

El programa:
  1) descarga velas intradía con yfinance;
  2) evita look-ahead: la señal usa solo velas ya cerradas;
  3) entra en la apertura de la vela siguiente;
  4) calcula retornos cortos brutos y netos de costes;
  5) compara los días señalados con todos los días;
  6) guarda tablas CSV y figuras PNG.

Uso básico:
    python iag_intraday_event_study.py

Dependencias:
    pip install yfinance pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Config:
    ticker: str = "IAG.MC"
    period: str = "60d"
    interval: str = "5m"
    timezone: str = "Europe/Madrid"
    session_start: str = "09:00"
    session_end: str = "17:30"
    checkpoints: tuple[str, ...] = ("09:15", "09:30", "10:00")
    exits: tuple[str, ...] = ("12:00", "15:00", "17:25")
    thresholds_pct: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    round_trip_cost_bps: float = 10.0
    bootstrap_samples: int = 10_000
    random_seed: int = 20260711
    min_complete_time: str = "17:25"
    chart_checkpoint: str = "09:30"
    chart_threshold_pct: float = 1.0
    chart_exit: str = "17:25"


def parse_clock(value: str) -> time:
    try:
        hh, mm = value.split(":")
        return time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Hora inválida: {value!r}; usa HH:MM") from exc


def minutes_from_midnight(value: str) -> int:
    t = parse_clock(value)
    return 60 * t.hour + t.minute


def interval_minutes(interval: str) -> int:
    if interval.endswith("m") and interval[:-1].isdigit():
        return int(interval[:-1])
    raise ValueError("Este script espera un intervalo expresado en minutos, p. ej. 5m.")


def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance puede devolver (campo, ticker) o (ticker, campo).
        if ticker in out.columns.get_level_values(-1):
            out.columns = out.columns.get_level_values(0)
        elif ticker in out.columns.get_level_values(0):
            out.columns = out.columns.get_level_values(-1)
        else:
            out.columns = ["_".join(map(str, c)).strip("_") for c in out.columns]
    out.columns = [str(c).strip().title() for c in out.columns]
    return out


def download_data(cfg: Config) -> pd.DataFrame:
    print(f"Descargando {cfg.ticker}: periodo={cfg.period}, intervalo={cfg.interval} ...")
    df = yf.download(
        cfg.ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        prepost=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(
            "La descarga no devolvió datos. Comprueba la conexión, el ticker y las "
            "limitaciones temporales de Yahoo/yfinance."
        )

    df = flatten_yfinance_columns(df, cfg.ticker)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas esenciales: {sorted(missing)}")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        # Normalmente Yahoo devuelve índice con zona horaria; si no, asumimos UTC.
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(cfg.timezone)
    df.index = idx

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    start_t = parse_clock(cfg.session_start)
    end_t = parse_clock(cfg.session_end)
    df = df[(df.index.time >= start_t) & (df.index.time <= end_t)].copy()
    if df.empty:
        raise RuntimeError("No quedan velas dentro del horario continuo de BME.")
    return df


def split_complete_sessions(df: pd.DataFrame, cfg: Config) -> dict[pd.Timestamp, pd.DataFrame]:
    sessions: dict[pd.Timestamp, pd.DataFrame] = {}
    min_end = parse_clock(cfg.min_complete_time)

    for session_date, group in df.groupby(df.index.normalize()):
        group = group.sort_index()
        if group.empty:
            continue
        # Elimina sesiones incompletas, especialmente el día actual si aún no cerró.
        if group.index[-1].time() < min_end:
            continue
        sessions[pd.Timestamp(session_date)] = group
    return sessions


def closed_bars_before(group: pd.DataFrame, clock: str, bar_minutes: int) -> pd.DataFrame:
    """Velas cuyo final ya ha ocurrido a la hora `clock`."""
    target_min = minutes_from_midnight(clock)
    starts = group.index.hour * 60 + group.index.minute
    ends = starts + bar_minutes
    return group[ends <= target_min]


def entry_bar_at_or_after(group: pd.DataFrame, clock: str) -> pd.Series | None:
    target_min = minutes_from_midnight(clock)
    starts = group.index.hour * 60 + group.index.minute
    candidates = group[starts >= target_min]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def exit_close_at(group: pd.DataFrame, clock: str, bar_minutes: int) -> tuple[float, pd.Timestamp] | None:
    closed = closed_bars_before(group, clock, bar_minutes)
    if closed.empty:
        return None
    row = closed.iloc[-1]
    return float(row["Close"]), pd.Timestamp(row.name)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    # Se procesa por bloques para no reservar matrices gigantes.
    means: list[np.ndarray] = []
    remaining = n_samples
    block = min(2_000, n_samples)
    while remaining > 0:
        b = min(block, remaining)
        idx = rng.integers(0, len(values), size=(b, len(values)))
        means.append(values[idx].mean(axis=1))
        remaining -= b
    boot = np.concatenate(means)
    return tuple(np.quantile(boot, [alpha / 2, 1 - alpha / 2]))  # type: ignore[return-value]


def build_trade_table(
    sessions: dict[pd.Timestamp, pd.DataFrame],
    cfg: Config,
) -> pd.DataFrame:
    bar_minutes = interval_minutes(cfg.interval)
    rows: list[dict[str, object]] = []
    prior_close: float | None = None

    for session_date in sorted(sessions):
        group = sessions[session_date]
        day_open = float(group.iloc[0]["Open"])
        day_close = float(group.iloc[-1]["Close"])
        gap_pct = math.nan if prior_close is None else 100.0 * (day_open / prior_close - 1.0)

        for checkpoint in cfg.checkpoints:
            observed = closed_bars_before(group, checkpoint, bar_minutes)
            entry_row = entry_bar_at_or_after(group, checkpoint)
            if observed.empty or entry_row is None:
                continue

            signal_price = float(observed.iloc[-1]["Close"])
            signal_pct = 100.0 * (signal_price / day_open - 1.0)
            entry_price = float(entry_row["Open"])
            entry_time = pd.Timestamp(entry_row.name)

            for exit_clock in cfg.exits:
                exit_info = exit_close_at(group, exit_clock, bar_minutes)
                if exit_info is None:
                    continue
                exit_price, exit_timestamp = exit_info
                if exit_timestamp <= entry_time:
                    continue

                path = group[(group.index >= entry_time) & (group.index <= exit_timestamp)]
                if path.empty:
                    continue

                gross_short = entry_price / exit_price - 1.0
                net_short = gross_short - cfg.round_trip_cost_bps / 10_000.0
                # Para una posición corta: favorable = caída; adverso = subida.
                mfe = entry_price / float(path["Low"].min()) - 1.0
                mae = entry_price / float(path["High"].max()) - 1.0

                rows.append(
                    {
                        "date": session_date.date().isoformat(),
                        "checkpoint": checkpoint,
                        "exit": exit_clock,
                        "day_open": day_open,
                        "day_close": day_close,
                        "gap_pct": gap_pct,
                        "signal_price": signal_price,
                        "signal_return_pct": signal_pct,
                        "entry_time": entry_time.isoformat(),
                        "entry_price": entry_price,
                        "exit_time": exit_timestamp.isoformat(),
                        "exit_price": exit_price,
                        "gross_short_return_pct": 100.0 * gross_short,
                        "net_short_return_pct": 100.0 * net_short,
                        "max_favorable_excursion_pct": 100.0 * mfe,
                        "max_adverse_excursion_pct": 100.0 * mae,
                        "day_volume": float(group["Volume"].sum()),
                    }
                )
        prior_close = day_close

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise RuntimeError("No se pudieron construir observaciones de entrada/salida.")
    return trades


def summarize_strategy(trades: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, object]] = []

    for checkpoint in cfg.checkpoints:
        for exit_clock in cfg.exits:
            base = trades[(trades["checkpoint"] == checkpoint) & (trades["exit"] == exit_clock)]
            if base.empty:
                continue
            all_returns = base["net_short_return_pct"].to_numpy(dtype=float)
            baseline_mean = float(np.mean(all_returns))

            for threshold in cfg.thresholds_pct:
                selected = base[base["signal_return_pct"] >= threshold]
                values = selected["net_short_return_pct"].to_numpy(dtype=float)
                n = len(values)
                if n == 0:
                    rows.append(
                        {
                            "checkpoint": checkpoint,
                            "threshold_pct": threshold,
                            "exit": exit_clock,
                            "n_events": 0,
                            "event_frequency_pct": 0.0,
                            "mean_net_short_pct": math.nan,
                            "median_net_short_pct": math.nan,
                            "win_rate_pct": math.nan,
                            "std_pct": math.nan,
                            "t_stat": math.nan,
                            "bootstrap_ci95_low_pct": math.nan,
                            "bootstrap_ci95_high_pct": math.nan,
                            "baseline_all_days_mean_pct": baseline_mean,
                            "edge_vs_all_days_pct": math.nan,
                            "compounded_return_pct": math.nan,
                            "mean_mfe_pct": math.nan,
                            "mean_mae_pct": math.nan,
                        }
                    )
                    continue

                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if n > 1 else math.nan
                t_stat = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else math.nan
                ci_low, ci_high = bootstrap_mean_ci(values, cfg.bootstrap_samples, rng)
                compounded = 100.0 * (np.prod(1.0 + values / 100.0) - 1.0)

                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "threshold_pct": threshold,
                        "exit": exit_clock,
                        "n_events": n,
                        "event_frequency_pct": 100.0 * n / len(base),
                        "mean_net_short_pct": mean,
                        "median_net_short_pct": float(np.median(values)),
                        "win_rate_pct": 100.0 * float(np.mean(values > 0)),
                        "std_pct": std,
                        "t_stat": t_stat,
                        "bootstrap_ci95_low_pct": ci_low,
                        "bootstrap_ci95_high_pct": ci_high,
                        "baseline_all_days_mean_pct": baseline_mean,
                        "edge_vs_all_days_pct": mean - baseline_mean,
                        "compounded_return_pct": compounded,
                        "mean_mfe_pct": float(selected["max_favorable_excursion_pct"].mean()),
                        "mean_mae_pct": float(selected["max_adverse_excursion_pct"].mean()),
                    }
                )

    return pd.DataFrame(rows).sort_values(["checkpoint", "threshold_pct", "exit"])


def normalized_intraday_paths(
    sessions: dict[pd.Timestamp, pd.DataFrame],
    trades: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_paths: list[pd.Series] = []
    event_paths: list[pd.Series] = []

    event_dates = set(
        trades[
            (trades["checkpoint"] == cfg.chart_checkpoint)
            & (trades["exit"] == cfg.chart_exit)
            & (trades["signal_return_pct"] >= cfg.chart_threshold_pct)
        ]["date"]
    )

    for session_date, group in sessions.items():
        if group.empty:
            continue
        norm = 100.0 * (group["Close"] / float(group.iloc[0]["Open"]) - 1.0)
        minute_index = group.index.hour * 60 + group.index.minute
        series = pd.Series(norm.to_numpy(), index=minute_index)
        # Si hubiera duplicados por cambios horarios, conserva el último.
        series = series.groupby(level=0).last()
        all_paths.append(series.rename(session_date.date().isoformat()))
        if session_date.date().isoformat() in event_dates:
            event_paths.append(series.rename(session_date.date().isoformat()))

    all_df = pd.concat(all_paths, axis=1).sort_index() if all_paths else pd.DataFrame()
    event_df = pd.concat(event_paths, axis=1).sort_index() if event_paths else pd.DataFrame()
    return all_df, event_df


def format_clock_axis(minutes: Iterable[int]) -> list[str]:
    return [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]


def create_figures(
    sessions: dict[pd.Timestamp, pd.DataFrame],
    trades: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> None:
    all_paths, event_paths = normalized_intraday_paths(sessions, trades, cfg)

    if not all_paths.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = all_paths.index.to_numpy()
        ax.plot(x, all_paths.mean(axis=1), label=f"Todos los días (N={all_paths.shape[1]})")
        if not event_paths.empty:
            ax.plot(
                event_paths.index.to_numpy(),
                event_paths.mean(axis=1),
                label=(
                    f"Subida ≥ {cfg.chart_threshold_pct:.1f}% a "
                    f"{cfg.chart_checkpoint} (N={event_paths.shape[1]})"
                ),
            )
        tick_positions = np.arange(minutes_from_midnight("09:00"), minutes_from_midnight("17:31"), 60)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(format_clock_axis(tick_positions), rotation=45)
        ax.axhline(0.0, linewidth=1)
        ax.axvline(minutes_from_midnight(cfg.chart_checkpoint), linestyle="--", linewidth=1)
        ax.set_xlabel("Hora de Madrid")
        ax.set_ylabel("Retorno desde la apertura (%)")
        ax.set_title(f"Trayectoria intradía media de {cfg.ticker}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "average_intraday_paths.png", dpi=180)
        plt.close(fig)

    selected = trades[
        (trades["checkpoint"] == cfg.chart_checkpoint)
        & (trades["exit"] == cfg.chart_exit)
        & (trades["signal_return_pct"] >= cfg.chart_threshold_pct)
    ]
    if not selected.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(selected["net_short_return_pct"], bins="auto", edgecolor="black")
        ax.axvline(0.0, linewidth=1)
        ax.axvline(selected["net_short_return_pct"].mean(), linestyle="--", linewidth=1)
        ax.set_xlabel("Retorno neto de la posición corta (%)")
        ax.set_ylabel("Número de sesiones")
        ax.set_title(
            f"Resultados: señal ≥ {cfg.chart_threshold_pct:.1f}% a {cfg.chart_checkpoint}, "
            f"salida {cfg.chart_exit}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "event_return_histogram.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest inicial del patrón intradía de IAG en BME")
    parser.add_argument("--ticker", default="IAG.MC")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Coste total ida+vuelta en puntos básicos (default: 10 bps = 0.10%%)",
    )
    parser.add_argument("--output", default="iag_intraday_results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        round_trip_cost_bps=args.cost_bps,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = download_data(cfg)
        raw.to_csv(output_dir / "iag_5min_raw.csv", index_label="datetime")

        sessions = split_complete_sessions(raw, cfg)
        if len(sessions) < 10:
            raise RuntimeError(f"Solo hay {len(sessions)} sesiones completas; son demasiado pocas.")

        trades = build_trade_table(sessions, cfg)
        summary = summarize_strategy(trades, cfg)

        trades.to_csv(output_dir / "all_candidate_trades.csv", index=False)
        summary.to_csv(output_dir / "strategy_summary.csv", index=False)

        chart_events = trades[
            (trades["checkpoint"] == cfg.chart_checkpoint)
            & (trades["exit"] == cfg.chart_exit)
            & (trades["signal_return_pct"] >= cfg.chart_threshold_pct)
        ].copy()
        chart_events.to_csv(output_dir / "default_strategy_event_days.csv", index=False)

        create_figures(sessions, trades, cfg, output_dir)
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)

        print("\nResumen principal:")
        primary = summary[
            (summary["checkpoint"] == cfg.chart_checkpoint)
            & (summary["threshold_pct"] == cfg.chart_threshold_pct)
            & (summary["exit"] == cfg.chart_exit)
        ]
        if primary.empty:
            print("No se pudo construir la configuración principal.")
        else:
            print(primary.to_string(index=False))

        print(f"\nSesiones completas analizadas: {len(sessions)}")
        print(f"Resultados guardados en: {output_dir.resolve()}")
        print("Interpretación prudente: 60 días sirven para explorar, no para validar una estrategia.")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
