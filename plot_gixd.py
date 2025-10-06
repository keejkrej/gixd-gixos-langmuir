from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import re

# from scipy.ndimage import median_filter
# from utils.math.com import exponential_weighted_center
from utils.fit.gixd import fit_mirrored_gaussian

# Fitting bounds for mirrored Gaussian parameters
# Format: (lower_bounds, upper_bounds) for (amplitude, center, sigma, offset)
AMPLITUDE_BOUNDS = (0, 10)
CENTER_BOUNDS = (0, 60)
SIGMA_BOUNDS = (0, 30)
OFFSET_BOUNDS = (-1, 5)

# Initial guess parameters for mirrored Gaussian fitting
# Format: (amplitude, center, sigma, offset)
MIRRORED_GAUSSIAN_INITIAL_GUESS = (2.0, 30.0, 20.0, 0.0)

PROCESSED_DIR = "processed/gixd"
PLOT_DIR = "plot/gixd"


def parse_index_pressure(filename_or_varname, is_water=False):
    if is_water:
        # For water: water_44_intensity_q.nc or water_44_cartesian.nc
        m = re.search(r"_(\d+)_", filename_or_varname)
        if m:
            idx = int(m.group(1))
            pressure = 0.0  # Water has pressure 0
            return idx, pressure

    # Regular samples: sample_78_10_intensity_q.nc or 78_10_sub_invquad_Iq
    # Try pattern with underscores first (for variable names)
    m = re.search(r"^(\d+)_([\d.]+|NA)_", filename_or_varname)
    if m:
        idx = int(m.group(1))
        pressure_str = m.group(2)
        pressure = float(pressure_str) if pressure_str != "NA" else "NA"
        return idx, pressure

    # Try pattern with sample name prefix (for filenames)
    m = re.search(r"_(\d+)_([\d.]+|NA)_", filename_or_varname)
    if m:
        idx = int(m.group(1))
        pressure_str = m.group(2)
        pressure = float(pressure_str) if pressure_str != "NA" else "NA"
        return idx, pressure

    # Try simple idx_pressure format (for horizontal slices): "78_5"
    m = re.search(r"^(\d+)_([\d.]+)$", filename_or_varname)
    if m:
        idx = int(m.group(1))
        pressure_str = m.group(2)
        pressure = float(pressure_str) if pressure_str != "NA" else "NA"
        return idx, pressure

    raise ValueError(f"Could not parse index and pressure from: {filename_or_varname}")


# Note: _plot_1d_from_dataset function removed as it's not used in current plotting pipeline


# Note: _plot_1d function removed as it's not used in current plotting pipeline


def plot_1d_profiles(sample_dir, plot_path, is_water=False):
    """Plot 1D profiles: sub_invquad_Iq, sub_invquad_Itau"""
    sample_name = sample_dir.name

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
        idx, pressure = parse_index_pressure(var_name, is_water)
        key = (idx, pressure)
        if key not in grouped_vars:
            grouped_vars[key] = {}
        grouped_vars[key][var_name] = ds_profiles[var_name]

    # Dictionary to store tau_max vs pressure for this sample
    tau_max_data = {}

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
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot I(q) profile
            if Iq_var is not None:
                ax1 = axes[0]
                q_vals = Iq_var["q"].values
                intensity_vals = Iq_var.values

                ax1.plot(q_vals, intensity_vals, "b-", linewidth=2, alpha=0.8)
                ax1.set_xlabel("q (Å$^{-1}$)")
                ax1.set_ylabel("Intensity (a.u.)")
                ax1.set_title("I(q) Profile")
                ax1.grid(True, alpha=0.3)

                # Store max position for tau_max_data
                tau_max_data[pressure] = q_vals[np.argmax(intensity_vals)]

            # Plot I(tau) profile
            if Itau_var is not None:
                ax2 = axes[1]
                tau_vals = Itau_var["tau"].values
                intensity_vals = Itau_var.values

                ax2.scatter(tau_vals, intensity_vals, color="red", s=20, alpha=0.7)
                ax2.set_xlabel("tau (deg)")
                ax2.set_ylabel("Intensity (a.u.)")
                ax2.set_title("I(τ) Profile")
                ax2.grid(True, alpha=0.3)

                # Fit mirrored Gaussian for tau profile
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
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    tau_max_data[pressure] = center_fit
                except Exception as e:
                    print(
                        f"Warning: Failed to fit mirrored Gaussian for tau profile at pressure {pressure}: {e}"
                    )
                    # Fallback to argmax
                    tau_max_data[pressure] = tau_vals[np.argmax(intensity_vals)]
                    ax2.axvline(
                        x=tau_max_data[pressure],
                        color="red",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                    )

            # Main title
            fig.suptitle(f"{sample_name} idx={idx}, p={pressure} mN/m", fontsize=14)
            plt.tight_layout()

            # Save the combined plot
            out_file = plot_path / f"{sample_name}_{idx}_{pressure}_1d_profiles.png"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return tau_max_data


# Note: plot_1d_profiles function removed as it's not used in current plotting pipeline


# Note: _plot_2d function removed as it's not used in current plotting pipeline


def plot_2d_maps(sample_dir, plot_path, is_water=False):
    """
    Plot 2D maps: orig_cart, bg_invquad_cart, sub_invquad_cart, sub_invquad_polar
    """
    sample_name = sample_dir.name

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
        idx, pressure = parse_index_pressure(var_name, is_water)
        key = (idx, pressure)
        if key not in grouped_vars:
            grouped_vars[key] = {}
        grouped_vars[key][var_name] = ds_maps[var_name]

    # Plot each group with essential maps only
    for (idx, pressure), vars_dict in grouped_vars.items():
        # Get essential data
        orig_cart = None
        bg_cart = None
        sub_cart = None
        sub_polar = None

        for var_name, da in vars_dict.items():
            if "_orig_cart" in var_name:
                orig_cart = da
            elif "_bg_invquad_cart" in var_name:
                bg_cart = da
            elif "_sub_invquad_cart" in var_name:
                sub_cart = da
            elif "_sub_invquad_polar" in var_name:
                sub_polar = da

        # Create a 2x2 subplot for the essential maps
        if orig_cart is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot original cartesian
            ax1 = axes[0, 0]
            im1 = ax1.imshow(
                orig_cart.values,
                origin="lower",
                extent=(
                    float(orig_cart["qxy"][0]),
                    float(orig_cart["qxy"][-1]),
                    float(orig_cart["qz"][0]),
                    float(orig_cart["qz"][-1]),
                ),
                aspect="auto",
            )
            ax1.set_xlabel("qxy (Å$^{-1}$)")
            ax1.set_ylabel("qz (Å$^{-1}$)")
            ax1.set_title("Original Data")
            fig.colorbar(im1, ax=ax1)

            # Plot background cartesian
            if bg_cart is not None:
                ax2 = axes[0, 1]
                im2 = ax2.imshow(
                    bg_cart.values,
                    origin="lower",
                    extent=(
                        float(bg_cart["qxy"][0]),
                        float(bg_cart["qxy"][-1]),
                        float(bg_cart["qz"][0]),
                        float(bg_cart["qz"][-1]),
                    ),
                    aspect="auto",
                )
                ax2.set_xlabel("qxy (Å$^{-1}$)")
                ax2.set_ylabel("qz (Å$^{-1}$)")
                ax2.set_title("Background")
                fig.colorbar(im2, ax=ax2)

            # Plot subtracted cartesian
            if sub_cart is not None:
                ax3 = axes[1, 0]
                im3 = ax3.imshow(
                    sub_cart.values,
                    origin="lower",
                    extent=(
                        float(sub_cart["qxy"][0]),
                        float(sub_cart["qxy"][-1]),
                        float(sub_cart["qz"][0]),
                        float(sub_cart["qz"][-1]),
                    ),
                    aspect="auto",
                )
                ax3.set_xlabel("qxy (Å$^{-1}$)")
                ax3.set_ylabel("qz (Å$^{-1}$)")
                ax3.set_title("Background Subtracted")
                fig.colorbar(im3, ax=ax3)

            # Plot polar
            if sub_polar is not None:
                ax4 = axes[1, 1]
                im4 = ax4.imshow(
                    sub_polar.values,
                    origin="lower",
                    extent=(
                        float(sub_polar["q"][0]),
                        float(sub_polar["q"][-1]),
                        float(sub_polar["tau"][0]),
                        float(sub_polar["tau"][-1]),
                    ),
                    aspect="auto",
                )
                ax4.set_xlabel("q (Å$^{-1}$)")
                ax4.set_ylabel("tau (deg)")
                ax4.set_title("Polar Coordinates")
                fig.colorbar(im4, ax=ax4)

            # Main title
            fig.suptitle(f"{sample_name} idx={idx}, p={pressure} mN/m", fontsize=14)
            plt.tight_layout()

            # Save the combined plot
            out_file = plot_path / f"{sample_name}_{idx}_{pressure}_2d_maps.png"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)


# Note: plot_2d_maps function removed as it's not used in current plotting pipeline


def main():
    processed_dir = Path(PROCESSED_DIR)
    plot_dir = Path(PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to collect tau_max vs pressure for all samples
    all_tau_max_data = {}

    for sample_dir in sorted(processed_dir.iterdir()):
        sample_name = sample_dir.name
        plot_path = plot_dir / sample_name
        plot_path.mkdir(parents=True, exist_ok=True)

        is_water = sample_name == "water"
        print(f"Plotting {sample_name}...")

        # Plot simplified set of plots
        if not is_water:
            tau_max_data = plot_1d_profiles(sample_dir, plot_path, is_water)
            plot_2d_maps(sample_dir, plot_path, is_water)
            plot_horizontal_slice_comparison(sample_dir, plot_path, is_water)

            # Store tau_max data for this sample (skip water for tau_max plot)
            if tau_max_data:
                all_tau_max_data[sample_name] = tau_max_data

    # Plot tau_max vs pressure for all samples
    if all_tau_max_data:
        fig_tau_max = plt.figure(figsize=(8, 8))
        ax_tau_max = fig_tau_max.add_subplot(111)

        for sample_name, tau_data in all_tau_max_data.items():
            # Filter out pressures < 5 only for dopc and redazo samples
            # if sample_name.lower() in ["dopc", "redazo"]:
            #     filtered_data = {p: tau_data[p] for p in tau_data.keys() if p >= 5}
            # else:
            #     filtered_data = tau_data
            # pressures = sorted(filtered_data.keys())
            # tau_max_values = [filtered_data[p] for p in pressures]
            pressures = sorted(tau_data.keys())
            tau_max_values = [tau_data[p] for p in pressures]

            if pressures:  # Only plot if there's data left after filtering
                ax_tau_max.plot(pressures, tau_max_values, "o-", label=sample_name)

        ax_tau_max.set_xlabel("Pressure (mN/m)")
        ax_tau_max.set_ylabel("Tau_center (deg)")
        ax_tau_max.set_title("Tau_center vs Pressure")
        ax_tau_max.legend()
        ax_tau_max.grid(True, alpha=0.3)

        plt.tight_layout()
        out_tau_max = plot_dir / "tau_max_vs_pressure.png"
        fig_tau_max.savefig(out_tau_max)
        plt.close(fig_tau_max)

    print("GIXD plotting completed.")


# Note: plot_background_files function removed as it's not used in current plotting pipeline


def plot_horizontal_slice_comparison(sample_dir, plot_path, is_water=False):
    """Plot horizontal slice profiles for original data, background, and difference."""
    sample_name = sample_dir.name

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
            # Extract idx_pressure from variable name like "78_5_horizontal_slices_original_qz_0.0_0.2"
            parts = var_name.split("_horizontal_slices_")
            if len(parts) == 2:
                idx_pressure = parts[0]
                slice_var = parts[1]
                if idx_pressure not in grouped_vars:
                    grouped_vars[idx_pressure] = {}
                grouped_vars[idx_pressure][slice_var] = ds[var_name]

        # Plot each idx_pressure group
        for idx_pressure, vars_dict in grouped_vars.items():
            idx, pressure = parse_index_pressure(idx_pressure, is_water)

            try:
                # Create plot with subplots for each qz range
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                # Get qz ranges from dataset attributes
                qz_ranges_str = ds.attrs.get(
                    "qz_ranges",
                    "[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]",
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
                    ]

                # Define colors for different qz ranges
                colors = ["blue", "green", "red", "orange", "purple"]

                for i, (qz_min, qz_max) in enumerate(qz_ranges):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    qz_label = f"{qz_min:.1f}_{qz_max:.1f}"

                    # Get the data for this qz range
                    orig_var = f"original_qz_{qz_label}"
                    bg_var = f"background_qz_{qz_label}"
                    diff_var = f"difference_qz_{qz_label}"

                    if (
                        orig_var not in vars_dict
                        or bg_var not in vars_dict
                        or diff_var not in vars_dict
                    ):
                        continue

                    orig_da = vars_dict[orig_var]
                    bg_da = vars_dict[bg_var]
                    diff_da = vars_dict[diff_var]

                    qxy_vals = orig_da["qxy"].values
                    orig_vals = orig_da.values
                    bg_vals = bg_da.values
                    diff_vals = diff_da.values

                    # Apply offset for better visualization
                    offset = i * 0.1 * np.max(orig_vals)

                    # Plot original data with offset
                    ax.plot(
                        qxy_vals,
                        orig_vals + offset,
                        color=colors[i],
                        linewidth=2,
                        label=f"Original (qz: {qz_min}-{qz_max})",
                        alpha=0.8,
                    )

                    # Plot background with offset
                    ax.plot(
                        qxy_vals,
                        bg_vals + offset,
                        color=colors[i],
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Background (qz: {qz_min}-{qz_max})",
                        alpha=0.6,
                    )

                    # Plot difference (residual) on secondary y-axis
                    ax2 = ax.twinx()
                    ax2.plot(
                        qxy_vals,
                        diff_vals,
                        color="gray",
                        linewidth=1,
                        alpha=0.5,
                    )

                    # Formatting
                    ax.set_xlabel("qxy (Å$^{-1}$)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.set_title(f"qz: {qz_min}-{qz_max}")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    # Hide secondary y-axis labels for cleaner look
                    ax2.set_ylabel("")
                    ax2.set_yticks([])

                # Hide unused subplots
                for i in range(len(qz_ranges), len(axes)):
                    axes[i].set_visible(False)

                # Main title
                fig.suptitle(
                    f"{sample_name} idx={idx}, p={pressure} mN/m\n"
                    f"Horizontal slice comparison (multiple qz ranges with offsets)",
                    fontsize=14,
                )

                plt.tight_layout()
                out_plot = (
                    plot_path / f"{sample_name}_{idx}_{pressure}_horizontal_slices.png"
                )
                fig.savefig(out_plot, dpi=150, bbox_inches="tight")
                plt.close(fig)

            except Exception as e:
                print(
                    f"Failed to plot horizontal slice comparison for {idx_pressure}: {e}"
                )

    except Exception as e:
        print(f"Failed to load horizontal slice files for {slice_file}: {e}")


if __name__ == "__main__":
    main()
