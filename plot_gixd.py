from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from utils.fit.gixd import fit_mirrored_gaussian
from data_gixd import get_samples

# Configure matplotlib for publication quality (from plot_paper.py)
plt.style.use("default")
rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Bitstream Vera Serif"],
        "font.sans-serif": ["DejaVu Sans", "Arial", "Bitstream Vera Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "text.usetex": False,  # Set to True if LaTeX is available
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


# Fitting bounds for mirrored Gaussian parameters
# Format: (lower_bounds, upper_bounds) for (amplitude, center, sigma, offset)
AMPLITUDE_BOUNDS = (0, 10)
CENTER_BOUNDS = (0, 50)
SIGMA_BOUNDS = (0, 30)
OFFSET_BOUNDS = (-1, 5)

# Initial guess parameters for mirrored Gaussian fitting
# Format: (amplitude, center, sigma, offset)
MIRRORED_GAUSSIAN_INITIAL_GUESS = (2.0, 30.0, 20.0, 0.0)

PROCESSED_DIR = "processed/gixd"
PLOT_DIR = "plot/gixd"

# Professional styling options (from plot_paper.py)
COLORMAP = "inferno"  # Professional colormap: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
COLORBAR_LABEL = "Intensity (a.u.)"
X_LABEL = r"$q_{xy}$ [Å$^{-1}$]"  # Using proper LaTeX-style formatting
Y_LABEL = r"$q_z$ [Å$^{-1}$]"  # Using proper LaTeX-style formatting
Q_LABEL = r"$q$ [Å$^{-1}$]"
TAU_LABEL = r"$\tau$ [deg]"
INTENSITY_LABEL = "Intensity (a.u.)"
FIGURE_SIZE_SINGLE = (4, 4)  # Standard single-column figure size
FIGURE_SIZE_DOUBLE = (8, 4)  # Side-by-side plots
FIGURE_SIZE_2X2 = (8, 8)  # 2x2 subplot layout


def get_sample_by_name(sample_name: str):
    """Get sample information by name."""
    samples = get_samples()
    for sample in samples:
        if sample["name"] == sample_name:
            return sample
    return None


def get_sample_data(sample_name: str):
    """Get sample information including index-pressure mappings."""
    sample = get_sample_by_name(sample_name)
    if sample:
        return dict(zip(sample["index"], sample["pressure"]))
    return {}


def get_index_pressure_from_sample(sample_name: str, idx: int):
    """Get pressure for a given sample and index from sample data."""
    sample = get_sample_by_name(sample_name)
    if not sample:
        return None

    try:
        idx_pos = sample["index"].index(idx)
        return sample["pressure"][idx_pos]
    except (ValueError, IndexError):
        return None


def extract_index_from_filename(filename_or_varname: str):
    """Extract index from filename by finding the first number."""
    # Split by common separators and look for numeric parts
    parts = filename_or_varname.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)

    # If no underscore-separated numbers, extract digits from any part
    for part in parts:
        digits = "".join(c for c in part if c.isdigit())
        if digits:
            return int(digits)

    return None


def parse_index_pressure_from_filename(filename_or_varname: str, sample_name: str):
    """Parse index and pressure using sample data instead of regex."""
    # For regular samples, extract index and look up pressure from sample data
    idx = extract_index_from_filename(filename_or_varname)
    if idx is not None:
        pressure = get_index_pressure_from_sample(sample_name, idx)
        if pressure is not None:
            return idx, pressure

    raise ValueError(f"Could not parse index and pressure from: {filename_or_varname}")


# Note: _plot_1d_from_dataset function removed as it's not used in current plotting pipeline


# Note: _plot_1d function removed as it's not used in current plotting pipeline


def get_sample_info(sample_name: str):
    """Get sample information including full_name."""
    sample = get_sample_by_name(sample_name)
    if sample:
        return sample["full_name"], sample["index"], sample["pressure"]
    return sample_name, [], []


def plot_1d_profiles(sample_dir, plot_path):
    """Plot 1D profiles: sub_invquad_Iq, sub_invquad_Itau with publication quality styling."""
    sample_name = sample_dir.name

    # Get full name for plot titles
    full_name, _, _ = get_sample_info(sample_name)

    # Find consolidated 1D profile file
    profile_file = sample_dir / f"{sample_name}_1d_profiles.nc"

    if not profile_file.exists():
        return {}

    # Load the consolidated dataset
    try:
        ds_profiles = xr.open_dataset(profile_file)
    except Exception as e:
        print(f"Failed to load consolidated 1D profiles for {sample_name}: {e}")
        return {}

    # Group variables by index and pressure
    grouped_vars = {}
    for var_name in ds_profiles.data_vars:
        idx, pressure = parse_index_pressure_from_filename(var_name, sample_name)
        key = (idx, pressure)
        if key not in grouped_vars:
            grouped_vars[key] = {}
        grouped_vars[key][var_name] = ds_profiles[var_name]

    # Create directories for individual plots and subplots
    individual_dir = plot_path / "individual" / "1d_profiles"
    subplot_dir = plot_path / "subplots" / "1d_profiles"
    comparison_dir = plot_path / "comparisons"

    for dir_path in [individual_dir, subplot_dir, comparison_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to store tau_max vs pressure for this sample as a DataFrame
    tilt_data = []

    # Store data for pressure comparisons
    iq_comparison_data = []
    itau_comparison_data = []

    # Plot each group with Iq and Itau side by side
    for (idx, pressure), vars_dict in grouped_vars.items():
        # Get Iq and Itau variables
        Iq_var = None
        Itau_var = None

        for var_name, da in vars_dict.items():
            if "_Iq" in var_name:
                Iq_var = da
            elif "_Itau" in var_name:
                Itau_var = da

        if Iq_var is not None or Itau_var is not None:
            # Create subplot with publication quality styling
            fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)

            # Plot I(q) profile
            if Iq_var is not None:
                ax1 = axes[0]
                q_vals = Iq_var["q"].values
                intensity_vals = Iq_var.values

                ax1.plot(q_vals, intensity_vals, "b-", linewidth=2, alpha=0.8)
                ax1.set_xlabel(Q_LABEL, fontsize=10)
                ax1.set_ylabel(INTENSITY_LABEL, fontsize=10)
                ax1.set_title("I(q) Profile", fontsize=12, style="italic")
                ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax1.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )

                # Store for comparison plot
                iq_comparison_data.append(
                    {
                        "idx": idx,
                        "pressure": pressure,
                        "q_vals": q_vals,
                        "intensity_vals": intensity_vals,
                    }
                )

                # Create individual Iq plot
                fig_iq, ax_iq = plt.subplots(1, 1, figsize=FIGURE_SIZE_SINGLE)
                ax_iq.plot(q_vals, intensity_vals, "b-", linewidth=2, alpha=0.8)
                ax_iq.set_xlabel(Q_LABEL, fontsize=10)
                ax_iq.set_ylabel(INTENSITY_LABEL, fontsize=10)
                ax_iq.set_title(
                    f"{full_name} I(q)\nidx={idx}, p={pressure} mN/m",
                    fontsize=12,
                    style="italic",
                )
                ax_iq.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax_iq.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )

                # Save individual Iq plot
                iq_file = individual_dir / f"{sample_name}_{idx}_{pressure}_Iq.png"
                fig_iq.savefig(iq_file, dpi=150, bbox_inches="tight")
                plt.close(fig_iq)

            # Plot I(tau) profile
            if Itau_var is not None:
                ax2 = axes[1]
                tau_vals = Itau_var["tau"].values
                intensity_vals = Itau_var.values

                ax2.scatter(tau_vals, intensity_vals, color="red", s=20, alpha=0.7)
                ax2.set_xlabel(TAU_LABEL, fontsize=10)
                ax2.set_ylabel(INTENSITY_LABEL, fontsize=10)
                ax2.set_title("I(τ) Profile", fontsize=12, style="italic")
                ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax2.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )

                # Fit mirrored Gaussian for tau profile
                center_fit = None
                fitted_curve = None

                # Store for comparison plot
                comparison_item = {
                    "idx": idx,
                    "pressure": pressure,
                    "tau_vals": tau_vals,
                    "intensity_vals": intensity_vals,
                    "center_fit": center_fit,  # Store the fitted center position
                    "fitted_curve": fitted_curve,  # Store the fitted curve
                }
                itau_comparison_data.append(comparison_item)

                try:
                    bounds = [
                        [
                            AMPLITUDE_BOUNDS[0],
                            CENTER_BOUNDS[0],
                            SIGMA_BOUNDS[0],
                            OFFSET_BOUNDS[0],
                        ],
                        [
                            AMPLITUDE_BOUNDS[1],
                            CENTER_BOUNDS[1],
                            SIGMA_BOUNDS[1],
                            OFFSET_BOUNDS[1],
                        ],
                    ]
                    amplitude_fit, center_fit, sigma_fit, offset_fit, fitted_curve = (
                        fit_mirrored_gaussian(
                            tau_vals,
                            intensity_vals,
                            initial_guess=MIRRORED_GAUSSIAN_INITIAL_GUESS,
                            bounds=bounds,
                        )
                    )
                    ax2.plot(tau_vals, fitted_curve, "r-", linewidth=2, alpha=1.0)
                    ax2.axvline(
                        x=center_fit,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    tilt_data.append({"pressure": pressure, "tilt": center_fit})

                    # Update the comparison data with the actual fitted results
                    comparison_item["center_fit"] = center_fit
                    comparison_item["fitted_curve"] = fitted_curve

                except Exception as e:
                    print(
                        f"Warning: Failed to fit mirrored Gaussian for tau profile at pressure {pressure}: {e}"
                    )
                    # Fallback to argmax
                    center_fit = tau_vals[np.argmax(intensity_vals)]
                    tilt_data.append({"pressure": pressure, "tilt": center_fit})
                    ax2.axvline(
                        x=center_fit,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )

                    # Update the comparison data with the fallback center
                    comparison_item["center_fit"] = center_fit

                # Create individual Itau plot
                fig_itau, ax_itau = plt.subplots(
                    1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
                )
                ax_itau.scatter(tau_vals, intensity_vals, color="red", s=20, alpha=0.7)
                ax_itau.set_xlabel(TAU_LABEL, fontsize=10)
                ax_itau.set_ylabel(INTENSITY_LABEL, fontsize=10)
                ax_itau.set_title(
                    f"{full_name} I(τ)\nidx={idx}, p={pressure} mN/m",
                    fontsize=12,
                    style="italic",
                )
                ax_itau.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax_itau.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )

                # Add fit if available
                try:
                    ax_itau.plot(tau_vals, fitted_curve, "r-", linewidth=2, alpha=1.0)
                    ax_itau.axvline(
                        x=center_fit,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                except Exception:
                    pass

                # Save individual Itau plot
                itau_file = individual_dir / f"{sample_name}_{idx}_{pressure}_Itau.png"
                fig_itau.savefig(itau_file, dpi=150, bbox_inches="tight")
                plt.close(fig_itau)

            # Main title
            fig.suptitle(
                f"{full_name} idx={idx}, p={pressure} mN/m",
                style="italic",
            )

            # Save the combined subplot
            subplot_file = (
                subplot_dir / f"{sample_name}_{idx}_{pressure}_1d_profiles.png"
            )
            fig.savefig(subplot_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Create pressure comparison plots
    create_pressure_comparison_plots(
        iq_comparison_data, itau_comparison_data, sample_name, comparison_dir
    )

    return tilt_data


# Note: plot_1d_profiles function removed as it's not used in current plotting pipeline


# Note: _plot_2d function removed as it's not used in current plotting pipeline


def plot_2d_maps(sample_dir, plot_path):
    """
    Plot 2D maps: raw_cart, bg_invquad_cart, sub_invquad_cart, sub_invquad_polar
    with publication quality styling.
    """
    sample_name = sample_dir.name

    # Get full name for plot titles
    full_name, _, _ = get_sample_info(sample_name)

    # Find organized 2D maps file
    maps_file = sample_dir / f"{sample_name}_2d_maps.nc"

    if not maps_file.exists():
        return

    # Load the organized dataset
    try:
        ds_maps = xr.open_dataset(maps_file)
    except Exception as e:
        print(f"Failed to load organized 2D maps for {sample_name}: {e}")
        return

    # Group variables by index and pressure
    grouped_vars = {}
    for var_name in ds_maps.data_vars:
        idx, pressure = parse_index_pressure_from_filename(var_name, sample_name)
        key = (idx, pressure)
        if key not in grouped_vars:
            grouped_vars[key] = {}
        grouped_vars[key][var_name] = ds_maps[var_name]

    # Create directories for individual plots and subplots
    individual_dir = plot_path / "individual" / "2d_maps"
    subplot_dir = plot_path / "subplots" / "2d_maps"

    for dir_path in [individual_dir, subplot_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Plot each group with essential maps only
    for (idx, pressure), vars_dict in grouped_vars.items():
        # Get essential data
        raw_cart = None
        bg_cart = None
        sub_cart = None
        sub_polar = None

        for var_name, da in vars_dict.items():
            if "_raw_cart" in var_name:
                raw_cart = da
            elif "_bg_invquad_cart" in var_name:
                bg_cart = da
            elif "_sub_invquad_cart" in var_name:
                sub_cart = da
            elif "_sub_invquad_polar" in var_name:
                sub_polar = da

        # Create a 2x2 subplot for the essential maps with publication quality styling
        if raw_cart is not None:
            fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_2X2)

            # Plot raw cartesian
            ax1 = axes[0, 0]
            data1 = raw_cart.values
            vmin1, vmax1 = (
                np.percentile(data1[~np.isnan(data1)], 1),
                np.percentile(data1[~np.isnan(data1)], 99),
            )
            im1 = ax1.imshow(
                data1,
                origin="lower",
                extent=(
                    float(raw_cart["qxy"][0]),
                    float(raw_cart["qxy"][-1]),
                    float(raw_cart["qz"][0]),
                    float(raw_cart["qz"][-1]),
                ),
                aspect="auto",
                cmap=COLORMAP,
                vmin=vmin1,
                vmax=vmax1,
                interpolation="nearest",
            )
            ax1.set_xlabel(X_LABEL, fontsize=10)
            ax1.set_ylabel(Y_LABEL, fontsize=10)
            ax1.set_title("Raw Data", fontsize=12, style="italic")
            ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax1.tick_params(
                axis="both", which="major", labelsize=9, width=0.8, length=4
            )
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label(COLORBAR_LABEL, fontsize=9)
            cbar1.ax.tick_params(labelsize=8)

            # Create individual plot for original cartesian
            fig_raw, ax_raw = plt.subplots(
                1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
            )
            im_raw = ax_raw.imshow(
                data1,
                origin="lower",
                extent=(
                    float(raw_cart["qxy"][0]),
                    float(raw_cart["qxy"][-1]),
                    float(raw_cart["qz"][0]),
                    float(raw_cart["qz"][-1]),
                ),
                aspect="auto",
                cmap=COLORMAP,
                vmin=vmin1,
                vmax=vmax1,
                interpolation="nearest",
            )
            ax_raw.set_xlabel(X_LABEL, fontsize=10)
            ax_raw.set_ylabel(Y_LABEL, fontsize=10)
            ax_raw.set_title(
                f"{full_name} Original\nidx={idx}, p={pressure} mN/m",
                fontsize=12,
                style="italic",
            )
            ax_raw.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax_raw.tick_params(
                axis="both", which="major", labelsize=9, width=0.8, length=4
            )
            cbar_raw = fig_raw.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
            cbar_raw.set_label(COLORBAR_LABEL, fontsize=9)
            cbar_raw.ax.tick_params(labelsize=8)

            # Save individual original plot
            raw_file = individual_dir / f"{sample_name}_{idx}_{pressure}_rawinal.png"
            fig_raw.savefig(raw_file, dpi=150, bbox_inches="tight")
            plt.close(fig_raw)

            # Plot background cartesian
            if bg_cart is not None:
                ax2 = axes[0, 1]
                data2 = bg_cart.values
                vmin2, vmax2 = (
                    np.percentile(data2[~np.isnan(data2)], 1),
                    np.percentile(data2[~np.isnan(data2)], 99),
                )
                im2 = ax2.imshow(
                    data2,
                    origin="lower",
                    extent=(
                        float(bg_cart["qxy"][0]),
                        float(bg_cart["qxy"][-1]),
                        float(bg_cart["qz"][0]),
                        float(bg_cart["qz"][-1]),
                    ),
                    aspect="auto",
                    cmap=COLORMAP,
                    vmin=vmin2,
                    vmax=vmax2,
                    interpolation="nearest",
                )
                ax2.set_xlabel(X_LABEL, fontsize=10)
                ax2.set_ylabel(Y_LABEL, fontsize=10)
                ax2.set_title("Background", fontsize=12, style="italic")
                ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax2.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )
                cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                cbar2.set_label(COLORBAR_LABEL, fontsize=9)
                cbar2.ax.tick_params(labelsize=8)

                # Create individual plot for background
                fig_bg, ax_bg = plt.subplots(
                    1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
                )
                im_bg = ax_bg.imshow(
                    data2,
                    origin="lower",
                    extent=(
                        float(bg_cart["qxy"][0]),
                        float(bg_cart["qxy"][-1]),
                        float(bg_cart["qz"][0]),
                        float(bg_cart["qz"][-1]),
                    ),
                    aspect="auto",
                    cmap=COLORMAP,
                    vmin=vmin2,
                    vmax=vmax2,
                    interpolation="nearest",
                )
                ax_bg.set_xlabel(X_LABEL, fontsize=10)
                ax_bg.set_ylabel(Y_LABEL, fontsize=10)
                ax_bg.set_title(
                    f"{full_name} Background\nidx={idx}, p={pressure} mN/m",
                    fontsize=12,
                    style="italic",
                )
                ax_bg.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax_bg.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )
                cbar_bg = fig_bg.colorbar(im_bg, ax=ax_bg, fraction=0.046, pad=0.04)
                cbar_bg.set_label(COLORBAR_LABEL, fontsize=9)
                cbar_bg.ax.tick_params(labelsize=8)

                # Save individual background plot
                bg_file = (
                    individual_dir / f"{sample_name}_{idx}_{pressure}_background.png"
                )
                fig_bg.savefig(bg_file, dpi=150, bbox_inches="tight")
                plt.close(fig_bg)

            # Plot subtracted cartesian
            if sub_cart is not None:
                ax3 = axes[1, 0]
                data3 = sub_cart.values
                vmin3, vmax3 = (
                    np.percentile(data3[~np.isnan(data3)], 1),
                    np.percentile(data3[~np.isnan(data3)], 99),
                )
                im3 = ax3.imshow(
                    data3,
                    origin="lower",
                    extent=(
                        float(sub_cart["qxy"][0]),
                        float(sub_cart["qxy"][-1]),
                        float(sub_cart["qz"][0]),
                        float(sub_cart["qz"][-1]),
                    ),
                    aspect="auto",
                    cmap=COLORMAP,
                    vmin=vmin3,
                    vmax=vmax3,
                    interpolation="nearest",
                )
                ax3.set_xlabel(X_LABEL, fontsize=10)
                ax3.set_ylabel(Y_LABEL, fontsize=10)
                ax3.set_title("Background Subtracted", fontsize=12, style="italic")
                ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax3.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )
                cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                cbar3.set_label(COLORBAR_LABEL, fontsize=9)
                cbar3.ax.tick_params(labelsize=8)

                # Create individual plot for subtracted cartesian
                fig_sub, ax_sub = plt.subplots(
                    1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
                )
                im_sub = ax_sub.imshow(
                    data3,
                    origin="lower",
                    extent=(
                        float(sub_cart["qxy"][0]),
                        float(sub_cart["qxy"][-1]),
                        float(sub_cart["qz"][0]),
                        float(sub_cart["qz"][-1]),
                    ),
                    aspect="auto",
                    cmap=COLORMAP,
                    vmin=vmin3,
                    vmax=vmax3,
                    interpolation="nearest",
                )
                ax_sub.set_xlabel(X_LABEL, fontsize=10)
                ax_sub.set_ylabel(Y_LABEL, fontsize=10)
                ax_sub.set_title(
                    f"{full_name} Background Subtracted\nidx={idx}, p={pressure} mN/m",
                    fontsize=12,
                    style="italic",
                )
                ax_sub.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax_sub.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )
                cbar_sub = fig_sub.colorbar(im_sub, ax=ax_sub, fraction=0.046, pad=0.04)
                cbar_sub.set_label(COLORBAR_LABEL, fontsize=9)
                cbar_sub.ax.tick_params(labelsize=8)

                # Save individual subtracted plot
                sub_file = (
                    individual_dir / f"{sample_name}_{idx}_{pressure}_subtracted.png"
                )
                fig_sub.savefig(sub_file, dpi=150, bbox_inches="tight")
                plt.close(fig_sub)

            # Plot polar
            if sub_polar is not None:
                ax4 = axes[1, 1]
                data4 = sub_polar.values
                vmin4, vmax4 = (
                    np.percentile(data4[~np.isnan(data4)], 1),
                    np.percentile(data4[~np.isnan(data4)], 99),
                )
                im4 = ax4.imshow(
                    data4,
                    origin="lower",
                    extent=(
                        float(sub_polar["q"][0]),
                        float(sub_polar["q"][-1]),
                        float(sub_polar["tau"][0]),
                        float(sub_polar["tau"][-1]),
                    ),
                    aspect="auto",
                    cmap=COLORMAP,
                    vmin=vmin4,
                    vmax=vmax4,
                    interpolation="nearest",
                )
                ax4.set_xlabel(Q_LABEL, fontsize=10)
                ax4.set_ylabel(TAU_LABEL, fontsize=10)
                ax4.set_title("Polar Coordinates", fontsize=12, style="italic")
                ax4.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax4.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )
                cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
                cbar4.set_label(COLORBAR_LABEL, fontsize=9)
                cbar4.ax.tick_params(labelsize=8)

                # Create individual plot for polar
                fig_pol, ax_pol = plt.subplots(
                    1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
                )
                im_pol = ax_pol.imshow(
                    data4,
                    origin="lower",
                    extent=(
                        float(sub_polar["q"][0]),
                        float(sub_polar["q"][-1]),
                        float(sub_polar["tau"][0]),
                        float(sub_polar["tau"][-1]),
                    ),
                    aspect="auto",
                    cmap=COLORMAP,
                    vmin=vmin4,
                    vmax=vmax4,
                    interpolation="nearest",
                )
                ax_pol.set_xlabel(Q_LABEL, fontsize=10)
                ax_pol.set_ylabel(TAU_LABEL, fontsize=10)
                ax_pol.set_title(
                    f"{full_name} Polar\nidx={idx}, p={pressure} mN/m",
                    fontsize=12,
                    style="italic",
                )
                ax_pol.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax_pol.tick_params(
                    axis="both", which="major", labelsize=9, width=0.8, length=4
                )
                cbar_pol = fig_pol.colorbar(im_pol, ax=ax_pol, fraction=0.046, pad=0.04)
                cbar_pol.set_label(COLORBAR_LABEL, fontsize=9)
                cbar_pol.ax.tick_params(labelsize=8)

                # Save individual polar plot
                pol_file = individual_dir / f"{sample_name}_{idx}_{pressure}_polar.png"
                fig_pol.savefig(pol_file, dpi=150, bbox_inches="tight")
                plt.close(fig_pol)

            # Main title
            fig.suptitle(
                f"{full_name} idx={idx}, p={pressure} mN/m",
                style="italic",
            )

            # Save the combined subplot
            subplot_file = subplot_dir / f"{sample_name}_{idx}_{pressure}_2d_maps.png"
            fig.savefig(subplot_file, dpi=150, bbox_inches="tight")
            plt.close(fig)


# Note: plot_2d_maps function removed as it's not used in current plotting pipeline


def create_pressure_comparison_plots(iq_data, itau_data, sample_name, comparison_dir):
    """Create comparison plots of different pressures for Iq and Itau."""

    # Get full name for plot titles
    full_name, _, _ = get_sample_info(sample_name)

    # Colors for different pressures - use unique pressures
    unique_pressures = sorted(
        set(item["pressure"] for item in iq_data)
        if iq_data
        else set(item["pressure"] for item in itau_data)
    )
    pressure_to_color = {
        pressure: plt.cm.viridis(i / len(unique_pressures))
        for i, pressure in enumerate(unique_pressures)
    }

    # Create I(q) pressure comparison plot
    if iq_data:
        fig_iq, ax_iq = plt.subplots(
            1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
        )

        for item in iq_data:
            pressure = item["pressure"]
            q_vals = item["q_vals"]
            intensity_vals = item["intensity_vals"]

            label = f"{pressure}"
            ax_iq.plot(
                q_vals,
                intensity_vals,
                color=pressure_to_color[pressure],
                linewidth=2,
                alpha=0.8,
                label=label,
            )

        ax_iq.set_xlabel(Q_LABEL, fontsize=10)
        ax_iq.set_ylabel(INTENSITY_LABEL, fontsize=10)
        ax_iq.set_title(
            f"{full_name} I(q)\nat various lateral pressures",
            fontsize=12,
            style="italic",
        )
        ax_iq.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax_iq.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=4)
        ax_iq.legend(fontsize=8, loc="best", title="p [mN/m]")

        # Save I(q) comparison plot
        iq_comp_file = comparison_dir / f"{sample_name}_Iq_pressure_comparison.png"
        fig_iq.savefig(
            iq_comp_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig_iq)

    # Create I(tau) pressure comparison plot
    if itau_data:
        fig_itau, ax_itau = plt.subplots(
            1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
        )

        for item in itau_data:
            pressure = item["pressure"]
            tau_vals = item["tau_vals"]
            intensity_vals = item["intensity_vals"]
            center_fit = item.get("center_fit")  # Get the fitted center position
            fitted_curve = item.get("fitted_curve")  # Get the fitted curve

            # Plot scatter points without legend
            ax_itau.scatter(
                tau_vals,
                intensity_vals,
                color=pressure_to_color[pressure],
                s=20,
                alpha=0.7,
                label="_nolegend_",  # Hide from legend
            )

            # Add fitted curve if available
            if fitted_curve is not None:
                ax_itau.plot(
                    tau_vals,
                    fitted_curve,
                    color=pressure_to_color[pressure],
                    linewidth=2,
                    alpha=1.0,
                    label=f"{pressure}",
                )

            # Add vertical line for center position if available
            if center_fit is not None:
                ax_itau.axvline(
                    x=center_fit,
                    color=pressure_to_color[pressure],
                    linestyle="-.",
                    linewidth=1.5,
                    alpha=0.7,
                )

        ax_itau.set_xlabel(TAU_LABEL, fontsize=10)
        ax_itau.set_ylabel(INTENSITY_LABEL, fontsize=10)
        ax_itau.set_title(
            f"{full_name} I(τ)\nat various lateral pressures",
            fontsize=12,
            style="italic",
        )
        ax_itau.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax_itau.tick_params(
            axis="both", which="major", labelsize=9, width=0.8, length=4
        )

        # Only add legend if there are items with valid labels
        if any(item.get("fitted_curve") is not None for item in itau_data):
            ax_itau.legend(fontsize=8, loc="best", title="p [mN/m]")

        # Save I(tau) comparison plot
        itau_comp_file = comparison_dir / f"{sample_name}_Itau_pressure_comparison.png"
        fig_itau.savefig(
            itau_comp_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig_itau)


# Note: plot_background_files function removed as it's not used in current plotting pipeline


def plot_horizontal_slice_comparison(sample_dir, plot_path):
    """Plot horizontal slice profiles for original data, background, and difference."""
    sample_name = sample_dir.name

    # Get full name for plot titles
    full_name, _, _ = get_sample_info(sample_name)

    # Find consolidated horizontal slice file
    slice_file = sample_dir / f"{sample_name}_horizontal_slices.nc"

    if not slice_file.exists():
        return

    try:
        # Load the consolidated Dataset containing all slices
        ds = xr.open_dataset(slice_file)

        # Group variables by idx_pressure
        grouped_vars = {}
        for var_name in ds.data_vars:
            # Extract idx_pressure from variable name like "78_5_horizontal_slices_rawinal_qz_0.0_0.2"
            parts = var_name.split("_horizontal_slices_")
            if len(parts) == 2:
                idx_pressure = parts[0]
                slice_var = parts[1]
                if idx_pressure not in grouped_vars:
                    grouped_vars[idx_pressure] = {}
                grouped_vars[idx_pressure][slice_var] = ds[var_name]

        # Plot each idx_pressure group
        for idx_pressure, vars_dict in grouped_vars.items():
            idx, pressure = parse_index_pressure_from_filename(
                idx_pressure, sample_name
            )

            try:
                # Create plot with subplots for each qz range
                fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
                axes = axes.flatten()

                # Get qz ranges from dataset attributes or use the same ranges as processing
                qz_ranges_str = ds.attrs.get(
                    "qz_ranges",
                    "[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2)]",
                )
                try:
                    qz_ranges = eval(qz_ranges_str)
                except (SyntaxError, ValueError, TypeError):
                    qz_ranges = [
                        (0.0, 0.2),
                        (0.2, 0.4),
                        (0.4, 0.6),
                        (0.6, 0.8),
                        (0.8, 1.0),
                        (1.0, 1.2),
                    ]

                # Define colors for different qz ranges
                colors = ["blue", "green", "red", "orange", "purple", "brown"]

                for i, (qz_min, qz_max) in enumerate(qz_ranges):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    qz_label = f"{qz_min:.1f}_{qz_max:.1f}"

                    # Get the data for this qz range
                    raw_var = f"original_qz_{qz_label}"
                    bg_var = f"background_qz_{qz_label}"
                    diff_var = f"difference_qz_{qz_label}"

                    if (
                        raw_var not in vars_dict
                        or bg_var not in vars_dict
                        or diff_var not in vars_dict
                    ):
                        continue

                    raw_da = vars_dict[raw_var]
                    bg_da = vars_dict[bg_var]
                    diff_da = vars_dict[diff_var]

                    qxy_vals = raw_da["qxy"].values
                    raw_vals = raw_da.values
                    bg_vals = bg_da.values
                    diff_vals = diff_da.values

                    # Plot original data
                    ax.plot(
                        qxy_vals,
                        raw_vals,
                        color=colors[i],
                        linewidth=2,
                        label="Raw",
                        alpha=0.8,
                    )

                    # Plot background
                    ax.plot(
                        qxy_vals,
                        bg_vals,
                        color=colors[i],
                        linestyle="--",
                        linewidth=1.5,
                        label="Background",
                        alpha=0.6,
                    )

                    # Plot difference (residual) on secondary y-axis
                    ax2 = ax.twinx()
                    ax2.plot(
                        qxy_vals,
                        diff_vals,
                        color="gray",
                        linewidth=2,
                        label="Difference",
                        alpha=0.7,
                    )

                    # Formatting with publication quality styling
                    ax.set_xlabel(X_LABEL, fontsize=10)
                    ax.set_ylabel(INTENSITY_LABEL, fontsize=10)
                    ax.set_title(f"qz: {qz_min}-{qz_max}", fontsize=12, style="italic")
                    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                    ax.tick_params(
                        axis="both", which="major", labelsize=9, width=0.8, length=4
                    )

                    # Show secondary y-axis labels for difference plot
                    ax2.set_ylabel("Difference (a.u.)", fontsize=10, color="gray")
                    ax2.tick_params(axis="y", labelsize=9, labelcolor="gray")

                    # Combine legends from both axes
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(
                        lines1 + lines2, labels1 + labels2, fontsize=8, loc="best"
                    )

                # Hide unused subplots
                for i in range(len(qz_ranges), len(axes)):
                    axes[i].set_visible(False)

                # Main title
                fig.suptitle(
                    f"{full_name} idx={idx}, p={pressure} mN/m\n"
                    f"Horizontal slice comparison (multiple qz ranges)",
                    style="italic",
                )

                # Layout already handled by constrained_layout=True
                slice_dir = plot_path / "horizontal_slices"
                slice_dir.mkdir(parents=True, exist_ok=True)
                out_plot = (
                    slice_dir / f"{sample_name}_{idx}_{pressure}_horizontal_slices.png"
                )
                fig.savefig(
                    out_plot,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                plt.close(fig)

            except Exception as e:
                print(
                    f"Failed to plot horizontal slice comparison for {idx_pressure}: {e}"
                )

    except Exception as e:
        print(f"Failed to load horizontal slice files for {slice_file}: {e}")


def main():
    processed_dir = Path(PROCESSED_DIR)
    plot_dir = Path(PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to collect tau_max vs pressure for all samples
    all_tilt_data = {}

    # Get samples to process
    samples = get_samples()
    sample_names = {sample["name"] for sample in samples}

    for sample_dir in sorted(processed_dir.iterdir()):
        sample_name = sample_dir.name

        # Only process samples that are in our sample list
        if sample_name not in sample_names:
            continue

        plot_path = plot_dir / sample_name
        plot_path.mkdir(parents=True, exist_ok=True)

        print(f"Plotting {sample_name}...")

        # Plot simplified set of plots
        tilt_data = plot_1d_profiles(
            sample_dir,
            plot_path,
        )
        plot_2d_maps(sample_dir, plot_path)
        plot_horizontal_slice_comparison(sample_dir, plot_path)

        # Store tau_max data for this sample - convert to DataFrame
        if tilt_data:
            tilt_df = pd.DataFrame(tilt_data)
            all_tilt_data[sample_name] = tilt_df

    # Plot tau_max vs pressure for all samples
    if all_tilt_data:
        fig_tilt, ax_tilt = plt.subplots(
            1, 1, figsize=FIGURE_SIZE_SINGLE, constrained_layout=True
        )

        for sample_name, tilt_df in all_tilt_data.items():
            # Get full name for legend
            full_name, _, _ = get_sample_info(sample_name)

            # Use pandas groupby to handle duplicates properly
            grouped = tilt_df.groupby("pressure")["tilt"].agg(["mean", "count"])
            pressures = grouped.index.values
            tau_values = grouped["mean"].values

            # Debug: print duplicate info
            duplicates = grouped[grouped["count"] > 1]
            if len(duplicates) > 0:
                print(f"{sample_name}: Found duplicate pressures:")
                for pressure, count in duplicates["count"].items():
                    print(f"  Pressure {pressure}: {count} entries")
                # Show all entries for debugging
                print(f"  All entries: {len(tilt_df)} total")
                print(tilt_df.sort_values("pressure").to_string())
            if sample_name == "dopc":
                color = "black"
            elif sample_name == "azotrans":
                color = "red"
            elif sample_name == "azocis":
                color = "blue"

            if len(pressures) > 0:  # Only plot if there's data left after filtering
                ax_tilt.plot(pressures, tau_values, "o-", label=full_name, color=color)

        ax_tilt.set_xlabel("Pressure [mN/m]")
        ax_tilt.set_ylabel("τ [deg]")
        ax_tilt.set_title(
            "Tilt angle (τ)\nat various lateral pressures", style="italic"
        )
        ax_tilt.legend(fontsize=8)
        ax_tilt.tick_params(
            axis="both", which="major", labelsize=9, width=0.8, length=4
        )
        ax_tilt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        out_tilt = plot_dir / "tilt_vs_pressure.png"
        fig_tilt.savefig(out_tilt)
        plt.close(fig_tilt)

    print("GIXD plotting completed.")


if __name__ == "__main__":
    main()
