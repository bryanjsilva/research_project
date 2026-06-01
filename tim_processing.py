"""Reusable analysis helpers for thermal interface material experiments.

The notebook can stay focused on the experimental story while this module
keeps the data loading, steady-state detection, calculations, aggregation, and
plot styling in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ACADEMIC_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#332288",  # indigo
    "#117733",  # dark green
)

MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def style_legend(legend: Optional[mpl.legend.Legend], alpha: float = 0.10) -> None:
    """Apply a subtle transparent legend background."""

    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(alpha)
    frame.set_edgecolor("#666666")


@dataclass(frozen=True)
class SampleConfig:
    """Metadata and processing options for one experimental sample."""

    sample_id: int
    sample: str
    specimen: int
    file_name: str
    label: str
    area_mm2: float
    condition: str = "Reference"
    skiprows: int = 8
    skip_time_min: float = 40.0
    delimiter: str = ";"
    decimal: str = ","
    time_column: str = "Scan Sweep Time (Sec)"
    temperature_token: str = "10"
    gauge_token: str = "20"
    force_calibration_n_per_v: float = 50_000.0
    bar_positions_m: Tuple[float, ...] = tuple(i * 3.5e-3 for i in range(9))
    hot_bar_position_indices: Tuple[int, ...] = (0, 1, 2, 3)
    cold_bar_position_indices: Tuple[int, ...] = (5, 6, 7, 8)
    hot_temperature_indices: Tuple[int, ...] = (0, 1, 2, 3)
    cold_temperature_indices: Tuple[int, ...] = (4, 5, 6, 7)
    excluded_temperature_indices: Tuple[int, ...] = ()
    interface_position_m: float = 14e-3
    active: bool = True


@dataclass(frozen=True)
class LoadedSample:
    """Dataframe plus inferred column groups for one sample."""

    config: SampleConfig
    frame: pd.DataFrame
    temperature_columns: List[str]
    gauge_columns: List[str]


def set_academic_style(font_size: int = 12) -> None:
    """Apply a clean, publication-friendly Matplotlib style."""

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "axes.prop_cycle": mpl.cycler(color=ACADEMIC_PALETTE),
        }
    )


def load_sample_data(
    config: SampleConfig,
    base_dir: Union[str, Path] = ".",
) -> LoadedSample:
    """Read a CSV file and infer temperature and gauge measurement columns."""

    csv_path = Path(base_dir) / config.file_name
    df = pd.read_csv(
        csv_path,
        delimiter=config.delimiter,
        decimal=config.decimal,
        skiprows=config.skiprows,
    )

    if config.time_column not in df.columns:
        available = ", ".join(map(str, df.columns[:8]))
        raise ValueError(
            f"Time column '{config.time_column}' was not found in {csv_path}. "
            f"First available columns: {available}"
        )

    extra_time_columns = [
        col for col in df.columns if "Time" in str(col) and col != config.time_column
    ]
    df = df.drop(columns=extra_time_columns)

    temperature_columns = [
        col for col in df.columns if config.temperature_token in str(col)
    ]
    gauge_columns = [col for col in df.columns if config.gauge_token in str(col)]

    if not temperature_columns:
        raise ValueError(
            f"No temperature columns found using token '{config.temperature_token}'."
        )
    if not gauge_columns:
        raise ValueError(f"No gauge columns found using token '{config.gauge_token}'.")

    df[config.time_column] = pd.to_datetime(df[config.time_column], errors="coerce")
    df = df.dropna(subset=[config.time_column]).set_index(config.time_column)
    df = df.sort_index()

    return LoadedSample(
        config=config,
        frame=df,
        temperature_columns=temperature_columns,
        gauge_columns=gauge_columns,
    )


def plot_temperature(
    sample: LoadedSample,
    steady_state: Optional[Sequence[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot all thermocouple signals for one sample."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.2))

    colors = plt.get_cmap("turbo_r")(np.linspace(0.05, 0.95, len(sample.temperature_columns)))
    excluded = set(sample.config.excluded_temperature_indices)
    for i, col in enumerate(sample.temperature_columns):
        is_excluded = i in excluded
        ax.plot(
            sample.frame.index,
            sample.frame[col],
            lw=1.0 if is_excluded else 1.2,
            color=colors[i],
            alpha=0.35 if is_excluded else 1.0,
            linestyle="--" if is_excluded else "-",
            label=f"TC {i + 1} (excluded)" if is_excluded else f"TC {i + 1}",
        )

    if steady_state:
        for i, (start, end) in enumerate(steady_state):
            label = "Steady-state window" if i == 0 else None
            ax.axvspan(start, end, color="#009E73", alpha=0.18, label=label)

    ax.set_title(sample.config.label)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Temperature ($^\circ$C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    style_legend(ax.legend(ncols=2, frameon=True))
    ax.grid(True, which="major")
    return ax


def plot_gauge_measurements(sample: LoadedSample) -> Tuple[plt.Figure, np.ndarray]:
    """Plot gauge channels in stacked axes with shared time axis."""

    n_axes = len(sample.gauge_columns)
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(8.0, max(2.4, 2.1 * n_axes)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, col in zip(axes, sample.gauge_columns):
        ax.plot(sample.frame.index, sample.frame[col], color="#332288", lw=1.1)
        ax.set_ylabel(col)
        ax.grid(True, which="major")

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle(f"Gauge measurements - {sample.config.label}", y=1.02)
    fig.tight_layout()
    return fig, axes


def find_steady_state(
    sample: LoadedSample,
    force_change: float = 0.5,
    steady_state_time_min: float = 5.0,
    max_variation: float = 0.35,
    skip_transition_min: float = 15.0,
    calculation_time_min: float = 5.0,
    plot: bool = True,
    verbose: bool = True,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Detect stable windows before load changes using smoothed temperatures."""

    config = sample.config
    df_temp = sample.frame
    temperature_columns = _valid_temperature_columns(sample)
    if len(temperature_columns) < 2:
        raise ValueError(
            "At least two valid thermocouples are required for steady-state detection. "
            "Check excluded_temperature_indices."
        )

    if verbose:
        print(f"{config.sample_id}: filtering noise and searching steady states")

    df_smooth = df_temp[temperature_columns].rolling(window="15s").mean().bfill()
    analysis_columns = list(df_smooth.columns[1:])

    zero_time = df_smooth.index[0]
    valid_time = zero_time + pd.Timedelta(minutes=config.skip_time_min)
    calculation_time = pd.Timedelta(minutes=calculation_time_min)

    shifted = df_smooth[analysis_columns].shift(freq=calculation_time)
    diff = (df_smooth[analysis_columns] - shifted.reindex(df_smooth.index, method="nearest")).abs()

    changes_mask = (diff >= force_change).any(axis="columns") & (
        df_smooth.index >= valid_time
    )
    all_changes = df_smooth.index[changes_mask]

    transition_spacing = pd.Timedelta(minutes=skip_transition_min)
    candidate_ends: List[pd.Timestamp] = []
    last_registered_time: Optional[pd.Timestamp] = None

    for timestamp in all_changes:
        if (
            last_registered_time is None
            or (timestamp - last_registered_time) >= transition_spacing
        ):
            candidate_ends.append(timestamp - pd.Timedelta(seconds=60))
            last_registered_time = timestamp

    steady_state: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    window_duration = pd.Timedelta(minutes=steady_state_time_min)

    for candidate_end in candidate_ends:
        candidate_start = candidate_end - window_duration
        window = df_smooth.loc[candidate_start:candidate_end, analysis_columns]
        if window.empty:
            continue

        temperature_range = window.max() - window.min()
        is_stable = (temperature_range <= max_variation).all()

        if verbose:
            status = "accepted" if is_stable else "rejected"
            variation = temperature_range.max()
            print(
                f"  {candidate_end:%H:%M:%S}: {status} "
                f"(max variation {variation:.3f} deg C)"
            )

        if is_stable:
            steady_state.append((candidate_start, candidate_end))

    if plot:
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        plot_temperature(sample, steady_state=steady_state, ax=ax)
        ax.axvline(valid_time, color="#666666", ls=":", lw=1.1, label="Ignored start")
        for i, (_, end) in enumerate(steady_state):
            label = "Load change" if i == 0 else None
            ax.axvline(end, color="#D55E00", ls="--", lw=1.1, label=label)
        style_legend(ax.legend(ncols=2, frameon=True))
        fig.tight_layout()

    return steady_state


def _select_by_indices(values: Sequence, indices: Sequence[int]) -> List:
    selected = []
    for i in indices:
        if i < 0:
            raise IndexError(
                f"Negative index {i} is not allowed for thermocouple selection."
            )
        if i >= len(values):
            raise IndexError(
                f"Index {i} is outside the available range 0-{len(values) - 1}. "
                f"Available values: {list(values)}"
            )
        selected.append(values[i])
    return selected


def _valid_temperature_columns(sample: LoadedSample) -> List[str]:
    """Return temperature columns that are not marked as defective."""

    excluded = set(sample.config.excluded_temperature_indices)
    return [
        column
        for index, column in enumerate(sample.temperature_columns)
        if index not in excluded
    ]


def _active_regression_inputs(
    sample: LoadedSample,
    position_indices: Sequence[int],
    temperature_indices: Sequence[int],
    bar_name: str,
) -> Tuple[np.ndarray, List[str], Tuple[int, ...]]:
    """Return positions and columns after excluding defective thermocouples."""

    config = sample.config
    if len(position_indices) != len(temperature_indices):
        raise ValueError(
            f"{bar_name} bar has {len(position_indices)} position indices but "
            f"{len(temperature_indices)} temperature indices."
        )

    excluded = set(config.excluded_temperature_indices)
    active_pairs = [
        (position_index, temperature_index)
        for position_index, temperature_index in zip(position_indices, temperature_indices)
        if temperature_index not in excluded
    ]

    if len(active_pairs) < 2:
        raise ValueError(
            f"{bar_name} bar needs at least two valid thermocouples for the "
            "linear regression. Check excluded_temperature_indices."
        )

    active_position_indices = tuple(position_index for position_index, _ in active_pairs)
    active_temperature_indices = tuple(temperature_index for _, temperature_index in active_pairs)

    positions = np.asarray(config.bar_positions_m, dtype=float)[list(active_position_indices)]
    columns = _select_by_indices(sample.temperature_columns, active_temperature_indices)
    return positions, columns, active_temperature_indices


def _safe_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def calculate_thermal_resistance(
    sample: LoadedSample,
    steady_state: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
    copper_conductivity_w_mk: float = 385.0,
    plot_regressions: bool = True,
) -> pd.DataFrame:
    """Calculate thermal resistance for all steady-state windows."""

    config = sample.config
    df = sample.frame
    results = []

    hot_x, hot_columns, active_hot_indices = _active_regression_inputs(
        sample,
        config.hot_bar_position_indices,
        config.hot_temperature_indices,
        "Hot",
    )
    cold_x, cold_columns, active_cold_indices = _active_regression_inputs(
        sample,
        config.cold_bar_position_indices,
        config.cold_temperature_indices,
        "Cold",
    )
    area_m2 = config.area_mm2 * 1e-6

    if len(hot_x) != len(hot_columns) or len(cold_x) != len(cold_columns):
        raise ValueError("Bar position indices and temperature columns are inconsistent.")

    for period_number, (start, end) in enumerate(steady_state, start=1):
        window = df.loc[start:end]
        hot_y = window[hot_columns].mean().to_numpy(dtype=float)
        cold_y = window[cold_columns].mean().to_numpy(dtype=float)
        gauge_means = window[sample.gauge_columns].mean()

        hot_slope, hot_intercept = np.polyfit(hot_x, hot_y, 1)
        cold_slope, cold_intercept = np.polyfit(cold_x, cold_y, 1)

        hot_fit = np.poly1d([hot_slope, hot_intercept])
        cold_fit = np.poly1d([cold_slope, cold_intercept])

        t_hot_interface = float(hot_fit(config.interface_position_m))
        t_cold_interface = float(cold_fit(config.interface_position_m))
        delta_t = t_hot_interface - t_cold_interface

        q_hot = -copper_conductivity_w_mk * hot_slope * area_m2
        q_cold = -copper_conductivity_w_mk * cold_slope * area_m2
        q_average = (q_hot + q_cold) / 2.0

        if np.isclose(q_average, 0.0):
            thermal_resistance = float("nan")
            specific_thermal_resistance = float("nan")
        else:
            thermal_resistance = delta_t / q_average
            specific_thermal_resistance = thermal_resistance * config.area_mm2 / 100.0

        gauge_voltage = float(gauge_means.iloc[0])
        gauge_temperature = float(gauge_means.iloc[1]) if len(gauge_means) > 1 else np.nan
        force_n = gauge_voltage * config.force_calibration_n_per_v
        pressure_mpa = force_n / config.area_mm2

        results.append(
            {
                "Sample_ID": config.sample_id,
                "Sample": config.sample,
                "Specimen": config.specimen,
                "Condition": config.condition,
                "Label": config.label,
                "Period": period_number,
                "Area_mm2": config.area_mm2,
                "Excluded_temperature_indices": tuple(config.excluded_temperature_indices),
                "Hot_temperature_indices_used": active_hot_indices,
                "Cold_temperature_indices_used": active_cold_indices,
                "Start": start,
                "End": end,
                "Gauge_voltage_V": gauge_voltage,
                "Gauge_temperature_C": gauge_temperature,
                "Force_N": force_n,
                "Pressure_MPa": pressure_mpa,
                "T_hot_interface_C": t_hot_interface,
                "T_cold_interface_C": t_cold_interface,
                "Delta_T_C": delta_t,
                "Q_hot_W": q_hot,
                "Q_cold_W": q_cold,
                "Q_average_W": q_average,
                "Thermal_resistance_K_W": thermal_resistance,
                "Specific_thermal_resistance_K_cm2_W": specific_thermal_resistance,
                "Hot_fit_R2": _safe_r_squared(hot_y, hot_fit(hot_x)),
                "Cold_fit_R2": _safe_r_squared(cold_y, cold_fit(cold_x)),
                # Backward-compatible aliases for old notebook cells.
                "Force": force_n,
                "R_th": thermal_resistance,
                "R_th_sp": specific_thermal_resistance,
            }
        )

        if plot_regressions:
            _plot_temperature_regression(
                config=config,
                period_number=period_number,
                hot_x=hot_x,
                cold_x=cold_x,
                hot_y=hot_y,
                cold_y=cold_y,
                hot_fit=hot_fit,
                cold_fit=cold_fit,
                t_hot_interface=t_hot_interface,
                t_cold_interface=t_cold_interface,
                q_hot=q_hot,
                q_cold=q_cold,
                q_average=q_average,
                specific_thermal_resistance=specific_thermal_resistance,
            )

    return pd.DataFrame(results)


def _plot_temperature_regression(
    config: SampleConfig,
    period_number: int,
    hot_x: np.ndarray,
    cold_x: np.ndarray,
    hot_y: np.ndarray,
    cold_y: np.ndarray,
    hot_fit: np.poly1d,
    cold_fit: np.poly1d,
    t_hot_interface: float,
    t_cold_interface: float,
    q_hot: float,
    q_cold: float,
    q_average: float,
    specific_thermal_resistance: float,
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    hot_color = "#D55E00"
    cold_color = "#0072B2"
    interface_color = "#117733"

    ax.plot(hot_x, hot_fit(hot_x), color=hot_color, lw=1.4, label="Hot bar fit")
    ax.plot(cold_x, cold_fit(cold_x), color=cold_color, lw=1.4, label="Cold bar fit")
    ax.scatter(hot_x, hot_y, color=hot_color, s=34, label="Hot bar data", zorder=3)
    ax.scatter(cold_x, cold_y, color=cold_color, s=34, label="Cold bar data", zorder=3)

    ax.scatter(
        config.interface_position_m,
        t_hot_interface,
        marker="x",
        color=hot_color,
        s=52,
        zorder=4,
        label=rf"$T_{{hot,int}}$ = {t_hot_interface:.2f} $^\circ$C",
    )
    ax.scatter(
        config.interface_position_m,
        t_cold_interface,
        marker="x",
        color=cold_color,
        s=52,
        zorder=4,
        label=rf"$T_{{cold,int}}$ = {t_cold_interface:.2f} $^\circ$C",
    )
    ax.axvline(
        config.interface_position_m,
        color=interface_color,
        lw=1.6,
        ls="--",
        label="Interface",
    )

    summary = (
        rf"$R_{{th,sp}}$ = {specific_thermal_resistance:.2f} "
        r"K cm$^2$ W$^{-1}$"
        "\n"
        rf"$Q_{{hot}}$ = {q_hot:.2f} W, "
        rf"$Q_{{cold}}$ = {q_cold:.2f} W"
        "\n"
        rf"$Q_{{avg}}$ = {q_average:.2f} W"
    )
    ax.text(
        0.03,
        0.04,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86},
    )

    formatter = ticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticks(config.bar_positions_m)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel(r"Temperature ($^\circ$C)")
    ax.set_title(f"{config.label} - period {period_number}")
    style_legend(ax.legend(frameon=True, fontsize=9, loc="best"))
    fig.tight_layout()
    return ax


def analyze_sample(
    config: SampleConfig,
    base_dir: Union[str, Path] = ".",
    plot_steady_state: bool = True,
    plot_regressions: bool = True,
    verbose: bool = True,
) -> Tuple[LoadedSample, List[Tuple[pd.Timestamp, pd.Timestamp]], pd.DataFrame]:
    """Load, detect steady states, and calculate resistance for one sample."""

    sample = load_sample_data(config, base_dir=base_dir)
    steady_state = find_steady_state(sample, plot=plot_steady_state, verbose=verbose)
    results = calculate_thermal_resistance(
        sample,
        steady_state,
        plot_regressions=plot_regressions,
    )
    return sample, steady_state, results


def combine_results(result_tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-sample result tables into one tidy dataframe."""

    tables = [table for table in result_tables if table is not None and not table.empty]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True).sort_values(
        ["Sample", "Condition", "Specimen", "Force_N"]
    )


def aggregate_results(
    results: pd.DataFrame,
    group_by: Sequence[str] = ("Sample", "Condition"),
    force_bin_width_n: Optional[float] = None,
    pressure_bin_width_mpa: Optional[float] = None,
    y_column: str = "Specific_thermal_resistance_K_cm2_W",
) -> pd.DataFrame:
    """Average results by sample/condition and optional force or pressure bins."""

    if results.empty:
        return pd.DataFrame()

    df = results.copy()
    keys = list(group_by)

    if force_bin_width_n is not None:
        df["Force_bin_N"] = (df["Force_N"] / force_bin_width_n).round() * force_bin_width_n
        keys.append("Force_bin_N")
    elif pressure_bin_width_mpa is not None:
        df["Pressure_bin_MPa"] = (
            df["Pressure_MPa"] / pressure_bin_width_mpa
        ).round() * pressure_bin_width_mpa
        keys.append("Pressure_bin_MPa")
    else:
        keys.append("Force_N")

    summary = (
        df.groupby(keys, dropna=False)
        .agg(
            n=("Sample_ID", "count"),
            Force_N=("Force_N", "mean"),
            Force_N_std=("Force_N", "std"),
            Pressure_MPa=("Pressure_MPa", "mean"),
            Pressure_MPa_std=("Pressure_MPa", "std"),
            Thermal_resistance_mean=(y_column, "mean"),
            Thermal_resistance_std=(y_column, "std"),
            Thermal_resistance_sem=(y_column, lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            Area_mm2=("Area_mm2", "mean"),
        )
        .reset_index()
        .sort_values(keys)
    )
    return summary


def exponential_decay_model(force_n: np.ndarray, amplitude: float, rate: float, offset: float) -> np.ndarray:
    """Monotonic decay model often suitable for resistance versus compression."""

    return amplitude * np.exp(-rate * force_n) + offset


def fit_exponential_decay(
    results: pd.DataFrame,
    y_column: str = "Specific_thermal_resistance_K_cm2_W",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fit a smooth exponential trendline, returning x/y arrays for plotting."""

    data = results[["Force_N", y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("Force_N")
    if len(data) < 3 or data["Force_N"].nunique() < 3:
        return None

    x = data["Force_N"].to_numpy(dtype=float)
    y = data[y_column].to_numpy(dtype=float)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    p0 = [y_span, 1.0 / max(float(np.nanmax(x) - np.nanmin(x)), 1.0), float(np.nanmin(y))]

    try:
        params, _ = curve_fit(
            exponential_decay_model,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    y_fit = exponential_decay_model(x_fit, *params)
    return x_fit, y_fit


def add_pressure_axis(
    ax: plt.Axes,
    area_mm2: float,
    label: Optional[str] = None,
) -> plt.Axes:
    """Add a top x-axis converting force to pressure for the selected area."""

    if area_mm2 <= 0:
        raise ValueError("The area for pressure conversion must be positive.")

    def force_to_pressure(force_n):
        return np.asarray(force_n) / area_mm2

    def pressure_to_force(pressure_mpa):
        return np.asarray(pressure_mpa) * area_mm2

    top_axis = ax.secondary_xaxis(
        "top", functions=(force_to_pressure, pressure_to_force)
    )
    top_axis.set_xlabel(label or "Pressure (MPa)")
    return top_axis


def plot_thermal_resistance_comparison(
    results: pd.DataFrame,
    group_by: str = "Label",
    y_column: str = "Specific_thermal_resistance_K_cm2_W",
    y_label: str = r"Specific thermal resistance, $R_{th,sp}$ (K cm$^2$ W$^{-1}$)",
    fit: bool = True,
    show_average: bool = False,
    average_by: Sequence[str] = ("Sample", "Condition"),
    force_bin_width_n: Optional[float] = None,
    mean_curve: bool = True,
    specimen_curve_points: bool = True,
    pressure_area_mm2: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    custom_color: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create the final force/resistance figure with pressure on the top axis."""

    if results.empty:
        raise ValueError("No results available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
    else:
        fig = ax.figure

    plot_individual_data = not (show_average and specimen_curve_points)
    grouped = results.sort_values("Force_N").groupby(group_by, sort=False)

    if plot_individual_data:
        for i, (label, group) in enumerate(grouped):
            color = custom_color or ACADEMIC_PALETTE[i % len(ACADEMIC_PALETTE)]
            marker = MARKERS[i % len(MARKERS)]
            ax.scatter(
                group["Force_N"],
                group[y_column],
                s=42,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.86,
                label=str(label),
                zorder=3,
            )

            if fit:
                fitted = fit_exponential_decay(group, y_column=y_column)
                if fitted is not None:
                    x_fit, y_fit = fitted
                    ax.plot(x_fit, y_fit, color=color, lw=1.4, ls="--", alpha=0.88)

    if show_average:
        averaged = aggregate_results(
            results,
            group_by=average_by,
            force_bin_width_n=force_bin_width_n,
            y_column=y_column,
        )
        if not averaged.empty:
            averaged["Average_label"] = averaged[list(average_by)].astype(str).agg(" / ".join, axis=1)
            for i, (label, group) in enumerate(averaged.groupby("Average_label", sort=False)):
                color = custom_color or ACADEMIC_PALETTE[i % len(ACADEMIC_PALETTE)]
                if specimen_curve_points:
                    ax.errorbar(
                        group["Force_N"],
                        group["Thermal_resistance_mean"],
                        yerr=group["Thermal_resistance_std"],
                        fmt="D",
                        ms=5.6,
                        mfc="white",
                        mec=color,
                        ecolor=color,
                        elinewidth=1.1,
                        capsize=3,
                        label=f"{label} mean",
                        zorder=4,
                    )

                if mean_curve:
                    mean_data = (
                        group[["Force_N", "Thermal_resistance_mean"]]
                        .rename(
                            columns={
                                "Thermal_resistance_mean": y_column,
                            }
                        )
                        .sort_values("Force_N")
                    )
                    fitted_mean = fit_exponential_decay(mean_data, y_column=y_column)
                    if fitted_mean is not None:
                        x_fit, y_fit = fitted_mean
                        ax.plot(
                            x_fit,
                            y_fit,
                            color=color,
                            lw=2.0,
                            ls="-",
                            alpha=0.95,
                            label=f"{label} mean curve",
                            zorder=5,
                        )
                    elif len(mean_data) >= 2:
                        ax.plot(
                            mean_data["Force_N"],
                            mean_data[y_column],
                            color=color,
                            lw=1.8,
                            ls="-",
                            alpha=0.88,
                            label=f"{label} mean curve",
                            zorder=5,
                        )

    if pressure_area_mm2 is None:
        areas = results["Area_mm2"].dropna().unique()
        if len(areas) == 1:
            pressure_area_mm2 = float(areas[0])
        else:
            pressure_area_mm2 = float(np.nanmedian(results["Area_mm2"]))
            warnings.warn(
                "Multiple contact areas are present. The top pressure axis uses "
                f"the median area ({pressure_area_mm2:g} mm^2). Pass "
                "pressure_area_mm2 explicitly for a different reference."
            )
            add_pressure_axis(
                ax,
                pressure_area_mm2,
                label=rf"Pressure (MPa, A = {pressure_area_mm2:g} mm$^2$)",
            )
            pressure_area_mm2 = None

    if pressure_area_mm2 is not None:
        add_pressure_axis(ax, pressure_area_mm2)

    ax.set_xlabel("Force (N)")
    ax.set_ylabel(y_label)
    ax.set_title("Thermal resistance as a function of compression")
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.13)
    style_legend(ax.legend(frameon=True, loc="best"))
    fig.tight_layout()
    return fig, ax


def academic_results_table(
    results: pd.DataFrame,
    include_measurements: bool = True,
    include_mean: bool = True,
    mean_group_by: Sequence[str] = ("Sample", "Condition"),
    force_bin_width_n: Optional[float] = None,
    pressure_bin_width_mpa: Optional[float] = None,
) -> pd.DataFrame:
    """Return a compact table with individual measurements and optional mean rows."""

    if results.empty:
        return pd.DataFrame()

    columns = [
        "Result_Type",
        "Sample_ID",
        "Sample",
        "Specimen",
        "Condition",
        "Period",
        "Excluded_temperature_indices",
        "Hot_temperature_indices_used",
        "Cold_temperature_indices_used",
        "n",
        "Force_N",
        "Force_N_std",
        "Pressure_MPa",
        "Pressure_MPa_std",
        "Q_average_W",
        "Delta_T_C",
        "Thermal_resistance_K_W",
        "Specific_thermal_resistance_K_cm2_W",
        "Specific_thermal_resistance_std",
        "Hot_fit_R2",
        "Cold_fit_R2",
    ]

    tables = []

    if include_measurements:
        individual = results.copy()
        individual["Result_Type"] = "Measurement"
        individual["Result_Order"] = 0
        individual["n"] = pd.NA
        individual["Force_N_std"] = pd.NA
        individual["Pressure_MPa_std"] = pd.NA
        individual["Specific_thermal_resistance_std"] = pd.NA
        tables.append(individual)

    if include_mean:
        averaged = aggregate_results(
            results,
            group_by=mean_group_by,
            force_bin_width_n=force_bin_width_n,
            pressure_bin_width_mpa=pressure_bin_width_mpa,
            y_column="Specific_thermal_resistance_K_cm2_W",
        )
        if not averaged.empty:
            mean_rows = averaged.copy()
            mean_rows["Result_Type"] = "Mean"
            mean_rows["Result_Order"] = 1
            mean_rows["Sample_ID"] = pd.NA
            mean_rows["Specimen"] = pd.NA
            mean_rows["Condition"] = mean_rows.get("Condition", pd.NA)
            mean_rows["Period"] = pd.NA
            mean_rows["Excluded_temperature_indices"] = pd.NA
            mean_rows["Hot_temperature_indices_used"] = pd.NA
            mean_rows["Cold_temperature_indices_used"] = pd.NA
            mean_rows["Q_average_W"] = pd.NA
            mean_rows["Delta_T_C"] = pd.NA
            mean_rows["Thermal_resistance_K_W"] = pd.NA
            mean_rows["Specific_thermal_resistance_K_cm2_W"] = mean_rows[
                "Thermal_resistance_mean"
            ]
            mean_rows["Specific_thermal_resistance_std"] = mean_rows[
                "Thermal_resistance_std"
            ]
            mean_rows["Hot_fit_R2"] = pd.NA
            mean_rows["Cold_fit_R2"] = pd.NA
            tables.append(mean_rows)

    if not tables:
        return pd.DataFrame(columns=columns)

    table = pd.concat(tables, ignore_index=True, sort=False)
    existing = [col for col in columns if col in table.columns]
    sort_columns = [
        col
        for col in ["Sample", "Condition", "Result_Order", "Force_N", "Specimen"]
        if col in table.columns
    ]
    if sort_columns:
        table = table.sort_values(sort_columns, na_position="last")
    return table[existing].round(4)
