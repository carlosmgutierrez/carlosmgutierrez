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
    round_trip_cost_bps: float = 10.0  # comisión ida+vuelta
    spread_bps: float = 3.0            # spread cruzado ida+vuelta (bid/ask)
    borrow_annual_pct: float = 3.0     # coste anual del préstamo de acciones (corto)
    stop_loss_pct: float = 0.0         # 0 = sin stop; p. ej. 1.0 = stop al +1% en contra
    bootstrap_samples: int = 10_000
    permutation_samples: int = 5_000
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
    return normalize_frame(df, cfg, assume_tz="UTC")


def normalize_frame(df: pd.DataFrame, cfg: Config, assume_tz: str) -> pd.DataFrame:
    """Valida columnas, ajusta zona horaria y recorta al horario de sesión."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas esenciales: {sorted(missing)}")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)
    idx = idx.tz_convert(cfg.timezone)
    df.index = idx

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    start_t = parse_clock(cfg.session_start)
    end_t = parse_clock(cfg.session_end)
    df = df[(df.index.time >= start_t) & (df.index.time <= end_t)].copy()
    if df.empty:
        raise RuntimeError("No quedan velas dentro del horario de sesión.")
    return df


def load_csv_data(cfg: Config, csv_path: str) -> pd.DataFrame:
    """Carga velas propias desde un CSV (para validar con años de historia).

    Formato esperado: una columna de fecha-hora (datetime/date/time/timestamp)
    y columnas Open, High, Low, Close, Volume (mayúsculas/minúsculas indistintas).
    Si la fecha-hora no lleva zona horaria se asume `cfg.timezone`.

    También admite el CSV tal cual lo exporta yfinance, con la cabecera de tres
    filas (Price / Ticker / Datetime).
    """
    print(f"Cargando velas desde CSV: {csv_path} ...")

    # Detecta el formato multi-cabecera de yfinance (filas Price / Ticker / ...).
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as handle:
        head = [handle.readline() for _ in range(2)]
    is_yahoo = (
        head[0].split(",")[0].strip() == "Price"
        and head[1].split(",")[0].strip() == "Ticker"
    )
    if is_yahoo:
        # Fila 0 = nombres de campo; filas 1-2 = Ticker y Datetime; datos desde la 3.
        df = pd.read_csv(csv_path, skiprows=[1, 2], index_col=0)
        df.index.name = "datetime"
        df = df.reset_index()
    else:
        df = pd.read_csv(csv_path)

    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    time_col = next((lower[k] for k in ("datetime", "date", "timestamp", "time") if k in lower), None)
    if time_col is None:
        raise RuntimeError("El CSV necesita una columna de fecha-hora (datetime/date/timestamp/time).")

    try:
        # Fechas tz-naive o con un único offset.
        parsed = pd.to_datetime(df[time_col], errors="coerce")
    except (ValueError, TypeError):
        # Offsets mixtos (p. ej. por el cambio de hora): normaliza a UTC.
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df.index = pd.DatetimeIndex(parsed)
    df = df.drop(columns=[time_col])
    df = df[df.index.notna()]
    if df.empty:
        raise RuntimeError("No se pudo interpretar ninguna fecha-hora del CSV.")

    df.columns = [str(c).strip().title() for c in df.columns]
    # Si la zona horaria no viene en los datos, se asume la de la configuración.
    return normalize_frame(df, cfg, assume_tz=cfg.timezone)


def merge_candles(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Fusiona dos conjuntos de velas por su índice temporal.

    En caso de solaparse una misma marca de tiempo, se conserva la fila más
    reciente (la de `new`). Sirve para acumular historia intradía ejecutando
    la descarga periódicamente (yfinance solo ofrece ~60 días de velas 5m).
    """
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


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


def permutation_pvalue(
    event: np.ndarray,
    other: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Contraste evento vs no-evento por permutación de etiquetas.

    Devuelve (diferencia_observada_de_medias, p_valor_bilateral). Prueba si el
    retorno de los días señalados difiere del de los NO señalados, que es la
    pregunta relevante, en lugar de compararlo solo contra cero.
    """
    event = event[np.isfinite(event)]
    other = other[np.isfinite(other)]
    if len(event) < 2 or len(other) < 2:
        return math.nan, math.nan
    obs = float(event.mean() - other.mean())
    pooled = np.concatenate([event, other])
    n_event = len(event)
    diffs = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:n_event].mean() - perm[n_event:].mean()
    p_value = float((np.abs(diffs) >= abs(obs) - 1e-12).mean())
    return obs, p_value


def equity_curve_metrics(returns_pct: np.ndarray) -> tuple[float, float]:
    """Drawdown máximo (%) y profit factor de una serie de retornos (%) en orden."""
    r = returns_pct[np.isfinite(returns_pct)] / 100.0
    if len(r) == 0:
        return math.nan, math.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    profit_factor = gains / losses if losses > 0 else math.inf
    return 100.0 * max_dd, profit_factor


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

                # Stop-loss para el corto: adverso = el precio SUBE. Si en alguna
                # vela del recorrido el High toca el nivel de stop, se sale ahí.
                exit_reason = "time"
                if cfg.stop_loss_pct and cfg.stop_loss_pct > 0.0:
                    stop_price = entry_price * (1.0 + cfg.stop_loss_pct / 100.0)
                    window = group[(group.index >= entry_time) & (group.index <= exit_timestamp)]
                    hit = window[window["High"] >= stop_price]
                    if not hit.empty:
                        exit_timestamp = pd.Timestamp(hit.index[0])
                        exit_price = stop_price
                        exit_reason = "stop"

                path = group[(group.index >= entry_time) & (group.index <= exit_timestamp)]
                if path.empty:
                    continue

                # Costes: comisión + spread cruzado + préstamo prorrateado por tiempo.
                holding_minutes = (exit_timestamp - entry_time).total_seconds() / 60.0
                commission = cfg.round_trip_cost_bps / 10_000.0
                spread = cfg.spread_bps / 10_000.0
                borrow = (cfg.borrow_annual_pct / 100.0) * (holding_minutes / 525_600.0)
                total_cost = commission + spread + borrow

                gross_short = entry_price / exit_price - 1.0
                net_short = gross_short - total_cost
                # Para una posición corta: favorable = caída; adverso = subida.
                mfe = entry_price / float(path["Low"].min()) - 1.0
                mae = entry_price / float(path["High"].max()) - 1.0

                rows.append(
                    {
                        "date": session_date.date().isoformat(),
                        "checkpoint": checkpoint,
                        "exit": exit_clock,
                        "exit_reason": exit_reason,
                        "day_open": day_open,
                        "day_close": day_close,
                        "gap_pct": gap_pct,
                        "signal_price": signal_price,
                        "signal_return_pct": signal_pct,
                        "entry_time": entry_time.isoformat(),
                        "entry_price": entry_price,
                        "exit_time": exit_timestamp.isoformat(),
                        "exit_price": exit_price,
                        "holding_minutes": holding_minutes,
                        "total_cost_bps": 10_000.0 * total_cost,
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

    empty_metrics = {
        "n_events": 0,
        "event_frequency_pct": 0.0,
        "n_stop_exits": 0,
        "mean_net_short_pct": math.nan,
        "median_net_short_pct": math.nan,
        "win_rate_pct": math.nan,
        "std_pct": math.nan,
        "sharpe_per_trade": math.nan,
        "t_stat": math.nan,
        "bootstrap_ci95_low_pct": math.nan,
        "bootstrap_ci95_high_pct": math.nan,
        "baseline_all_days_mean_pct": math.nan,
        "edge_vs_all_days_pct": math.nan,
        "nonevent_mean_pct": math.nan,
        "event_minus_nonevent_pct": math.nan,
        "perm_pvalue": math.nan,
        "compounded_return_pct": math.nan,
        "max_drawdown_pct": math.nan,
        "profit_factor": math.nan,
        "mean_mfe_pct": math.nan,
        "mean_mae_pct": math.nan,
    }

    for checkpoint in cfg.checkpoints:
        for exit_clock in cfg.exits:
            base = trades[(trades["checkpoint"] == checkpoint) & (trades["exit"] == exit_clock)]
            if base.empty:
                continue
            base = base.sort_values("date")
            all_returns = base["net_short_return_pct"].to_numpy(dtype=float)
            baseline_mean = float(np.mean(all_returns))

            for threshold in cfg.thresholds_pct:
                selected = base[base["signal_return_pct"] >= threshold]
                non_event = base[base["signal_return_pct"] < threshold]
                values = selected["net_short_return_pct"].to_numpy(dtype=float)
                other = non_event["net_short_return_pct"].to_numpy(dtype=float)
                n = len(values)

                row = {"checkpoint": checkpoint, "threshold_pct": threshold, "exit": exit_clock}
                row.update(empty_metrics)
                row["baseline_all_days_mean_pct"] = baseline_mean
                if n == 0:
                    rows.append(row)
                    continue

                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if n > 1 else math.nan
                t_stat = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else math.nan
                ci_low, ci_high = bootstrap_mean_ci(values, cfg.bootstrap_samples, rng)
                compounded = 100.0 * (np.prod(1.0 + values / 100.0) - 1.0)
                obs_diff, perm_p = permutation_pvalue(values, other, cfg.permutation_samples, rng)
                max_dd, profit_factor = equity_curve_metrics(values)

                row.update(
                    {
                        "n_events": n,
                        "event_frequency_pct": 100.0 * n / len(base),
                        "n_stop_exits": int((selected["exit_reason"] == "stop").sum()),
                        "mean_net_short_pct": mean,
                        "median_net_short_pct": float(np.median(values)),
                        "win_rate_pct": 100.0 * float(np.mean(values > 0)),
                        "std_pct": std,
                        "sharpe_per_trade": mean / std if (n > 1 and std > 0) else math.nan,
                        "t_stat": t_stat,
                        "bootstrap_ci95_low_pct": ci_low,
                        "bootstrap_ci95_high_pct": ci_high,
                        "edge_vs_all_days_pct": mean - baseline_mean,
                        "nonevent_mean_pct": float(np.mean(other)) if len(other) else math.nan,
                        "event_minus_nonevent_pct": obs_diff,
                        "perm_pvalue": perm_p,
                        "compounded_return_pct": compounded,
                        "max_drawdown_pct": max_dd,
                        "profit_factor": profit_factor,
                        "mean_mfe_pct": float(selected["max_favorable_excursion_pct"].mean()),
                        "mean_mae_pct": float(selected["max_adverse_excursion_pct"].mean()),
                    }
                )
                rows.append(row)

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
        help="Comisión ida+vuelta en puntos básicos (default: 10 bps = 0.10%%)",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=3.0,
        help="Spread bid/ask cruzado ida+vuelta en bps (default: 3)",
    )
    parser.add_argument(
        "--borrow-annual-pct",
        type=float,
        default=3.0,
        help="Coste anual del préstamo de acciones para el corto (default: 3%%)",
    )
    parser.add_argument(
        "--stop-pct",
        type=float,
        default=0.0,
        help="Stop-loss del corto en %% en contra (0 = sin stop; p. ej. 1.0)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Ruta a un CSV propio de velas (evita el límite de 60 días de yfinance)",
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
        spread_bps=args.spread_bps,
        borrow_annual_pct=args.borrow_annual_pct,
        stop_loss_pct=args.stop_pct,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = load_csv_data(cfg, args.csv) if args.csv else download_data(cfg)
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
            key_cols = [
                "checkpoint", "threshold_pct", "exit", "n_events", "mean_net_short_pct",
                "win_rate_pct", "sharpe_per_trade", "event_minus_nonevent_pct",
                "perm_pvalue", "max_drawdown_pct", "profit_factor",
            ]
            print(primary[key_cols].to_string(index=False))

        print(f"\nSesiones completas analizadas: {len(sessions)}")
        print(f"Resultados guardados en: {output_dir.resolve()}")
        print(
            "Aviso: se prueban 36 configuraciones; 'perm_pvalue' NO está corregido por "
            "comparaciones múltiples. Fija UNA hipótesis y valida fuera de muestra."
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
