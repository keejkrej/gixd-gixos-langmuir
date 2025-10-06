from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.ma as ma
import glob
import re
from matplotlib.colors import ListedColormap

# from scipy.ndimage import median_filter
# from utils.math.com import exponential_weighted_center
from utils.fit.gixd import fit_mirrored_gaussian
from data_gixd import ROI_IQ, ROI_ITAU

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


def parse_index_pressure(filename, is_water=False):
    if is_water:
        # For water: water_44_intensity_q.nc or water_44_cartesian.nc
        m = re.search(r"_(\d+)_", filename)
        if m:
            idx = int(m.group(1))
            pressure = 0.0  # Water has pressure 0
            return idx, pressure

    # Regular samples: sample_78_10_intensity_q.nc
    m = re.search(r"_(\d+)_([\d.]+|NA)_", filename)
    if not m:
        raise ValueError(
            f"Could not parse index and pressure from filename: {filename}"
        )
    idx = int(m.group(1))
    pressure_str = m.group(2)
    pressure = float(pressure_str) if pressure_str != "NA" else "NA"
    return idx, pressure


def _plot_1d(
    file_info: list,
    axis_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    tau_max_data: dict,
):
    """Shared helper to plot 1-D intensity profiles.

    Parameters
    ----------
    file_info: list of tuples (filepath, pressure, idx)
    axis_name: "q" or "tau" - the coordinate dimension in the DataArray.
    xlabel, ylabel, title: Plot labels.
    out_path: Destination path for the saved PNG.
    tau_max_data: Dictionary that will be populated with pressure -> argmax of the
        intensity for each file (mirrors the original implementation).
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Extract pressures for colormap normalization
    pressures = [pressure for _, pressure, _ in file_info if pressure != "NA"]
    if pressures:
        # Create colormap for pressures
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(pressures), vmax=max(pressures))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    for i, (f, pressure, idx) in enumerate(file_info):
        da = xr.open_dataarray(f)
        intensity = da.values
        axis_vals = da[axis_name].values

        # Label suffix distinguishes original vs water-subtracted vs invquad-subtracted data
        suffix = ""
        if "_orig_" in f:
            suffix = "orig"
        elif "_sub_water_" in f:
            suffix = "sub_water"
        elif "_sub_invquad_" in f:
            suffix = "sub_invquad"
        elif "_sub_" in f:
            # Legacy support for old naming convention
            suffix = "sub"
        label = f"idx={idx}, p={pressure}[mN/m]" + (f" ({suffix})" if suffix else "")

        bounds = [
            [AMPLITUDE_BOUNDS[0], CENTER_BOUNDS[0], SIGMA_BOUNDS[0], OFFSET_BOUNDS[0]],
            [AMPLITUDE_BOUNDS[1], CENTER_BOUNDS[1], SIGMA_BOUNDS[1], OFFSET_BOUNDS[1]],
        ]
        # For tau profiles, fit mirrored Gaussian and update label with parameters
        if axis_name == "tau":
            try:
                amplitude_fit, center_fit, sigma_fit, offset_fit, fitted_curve = (
                    fit_mirrored_gaussian(
                        axis_vals,
                        intensity,
                        initial_guess=MIRRORED_GAUSSIAN_INITIAL_GUESS,
                        bounds=bounds,
                    )
                )
                # Update label to include fitted parameters
                label += f"\nA={amplitude_fit:.2f}, center={center_fit:.2f}, σ={sigma_fit:.2f}, offset={offset_fit:.2f}"
            except Exception as e:
                print(
                    f"Warning: Failed to fit mirrored Gaussian for tau profile at pressure {pressure}: {e}"
                )
                # Add fallback note to label
                label += "\n(fit failed, using argmax)"

        # Set color based on pressure, style based on data type
        if pressure == "NA":
            color = "gray"
        else:
            color = cmap(norm(pressure))

        if axis_name == "q":
            # Use line plot for q profiles
            ax.plot(
                axis_vals,
                intensity,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=label,
            )
        else:
            # Keep scatter plot for tau profiles
            ax.scatter(
                axis_vals,
                intensity,
                color=color,
                s=20,  # marker size
                alpha=0.7,
                label=label,
            )

        # For tau profiles, also plot the fitted mirrored Gaussian curve and center line
        if axis_name == "tau":
            try:
                amplitude_fit, center_fit, sigma_fit, offset_fit, fitted_curve = (
                    fit_mirrored_gaussian(
                        axis_vals,
                        intensity,
                        initial_guess=MIRRORED_GAUSSIAN_INITIAL_GUESS,
                        bounds=bounds,
                    )
                )
                ax.plot(
                    axis_vals,
                    fitted_curve,
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    alpha=1.0,
                )
                # Add vertical line at the fitted center position
                ax.axvline(
                    x=center_fit,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                )
                tau_max_data[pressure] = center_fit
            except Exception as e:
                print(
                    f"Warning: Failed to fit mirrored Gaussian for tau profile at pressure {pressure}: {e}"
                )
                # Fallback to argmax if fitting fails
                center_fallback = axis_vals[np.argmax(intensity)]
                tau_max_data[pressure] = center_fallback
                # Still add a vertical line at the fallback position
                ax.axvline(
                    x=center_fallback,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                )
        else:
            # For q profiles, just use the position of maximum intensity
            tau_max_data[pressure] = axis_vals[np.argmax(intensity)]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if axis_name == "tau":
        title += " (data points with mirrored Gaussian fits & centers)"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    # Add colorbar if we have pressure data
    if pressures:
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Pressure (mN/m)")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_1d_profiles(sample_dir, plot_path, is_water=False):
    sample_name = sample_dir.name

    # Find all 1D profile files
    # I(q) profiles
    Iq_files_sub_water = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_sub_water_Iq.nc"))
    )
    Iq_files_sub_invquad = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_sub_invquad_Iq.nc"))
    )
    Iq_files = Iq_files_sub_water + Iq_files_sub_invquad

    # I(tau) profiles
    Itau_files_sub_water = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_sub_water_Itau.nc"))
    )
    Itau_files_sub_invquad = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_sub_invquad_Itau.nc"))
    )
    Itau_files = Itau_files_sub_water + Itau_files_sub_invquad

    # Dictionary to store tau_max vs pressure for this sample
    tau_max_data = {}

    # Plot I(q) profiles
    if Iq_files:
        Iq_file_info = []
        for f in Iq_files:
            idx, pressure = parse_index_pressure(f, is_water)
            Iq_file_info.append((f, pressure, idx))
        Iq_file_info.sort(key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2]))

        out_Iq = plot_path / f"{sample_name}_Iq_profiles.png"
        _plot_1d(
            Iq_file_info,
            axis_name="q",
            xlabel="q (Å$^{-1}$)",
            ylabel="Intensity (a.u.)",
            title=f"{sample_name} - I(q) Profiles",
            out_path=out_Iq,
            tau_max_data=tau_max_data,
        )

    # Plot I(tau) profiles
    if Itau_files:
        Itau_file_info = []
        for f in Itau_files:
            idx, pressure = parse_index_pressure(f, is_water)
            Itau_file_info.append((f, pressure, idx))
        Itau_file_info.sort(
            key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2])
        )

        out_Itau = plot_path / f"{sample_name}_Itau_profiles.png"
        _plot_1d(
            Itau_file_info,
            axis_name="tau",
            xlabel="tau (deg)",
            ylabel="Intensity (a.u.)",
            title=f"{sample_name} - I(τ) Profiles",
            out_path=out_Itau,
            tau_max_data=tau_max_data,
        )

    return tau_max_data


def _plot_2d(
    cart_da: xr.DataArray,
    polar_da: xr.DataArray | None,
    sample_name: str,
    idx: int,
    pressure: float,
    suffix: str,
    plot_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_da: xr.DataArray | None = None,
):
    """Helper to generate 2D visualizations for a single frame."""
    # Plot I(qxy, qz) map
    fig_cart = plt.figure(figsize=(8, 8))
    ax_cart = fig_cart.add_subplot(111)

    im_cart = ax_cart.imshow(
        cart_da.values,
        origin="lower",
        extent=(
            float(cart_da["qxy"][0]),
            float(cart_da["qxy"][-1]),
            float(cart_da["qz"][0]),
            float(cart_da["qz"][-1]),
        ),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax_cart.set_xlabel("qxy (A$^{-1}$)")
    ax_cart.set_ylabel("qz (A$^{-1}$)")
    title_suffix = f" ({suffix})"
    if mask_da is not None:
        title_suffix += " with fit points"
    ax_cart.set_title(
        f"{sample_name} idx={idx}, p={pressure}[mN/m]{title_suffix} - I(qxy, qz)"
    )
    fig_cart.colorbar(im_cart, ax=ax_cart)

    # Overlay fit mask if provided
    if mask_da is not None:
        # Create a masked array to show only True regions as solid red
        masked_mask = ma.masked_where(mask_da.values == 0, np.ones_like(mask_da.values))
        red_cmap = ListedColormap(["red"])
        ax_cart.imshow(
            masked_mask,
            extent=(
                float(mask_da["qxy"][0]),
                float(mask_da["qxy"][-1]),
                float(mask_da["qz"][0]),
                float(mask_da["qz"][-1]),
            ),
            origin="lower",
            alpha=0.5,
            cmap=red_cmap,
            zorder=10,
        )

    plt.tight_layout()
    out_cart = plot_path / f"{sample_name}_{idx}_{pressure}_{suffix}_cart.png"
    fig_cart.savefig(out_cart)
    plt.close(fig_cart)

    # Only plot polar if polar_da is provided (skip for orig data)
    if polar_da is None:
        return

    # Plot I(q, τ) map
    fig_polar = plt.figure(figsize=(8, 8))
    ax_polar = fig_polar.add_subplot(111)

    im_polar = ax_polar.imshow(
        polar_da.values,
        origin="lower",
        extent=(
            float(polar_da["q"][0]),
            float(polar_da["q"][-1]),
            float(polar_da["tau"][0]),
            float(polar_da["tau"][-1]),
        ),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Draw ROI_Q rectangle
    rect_q = patches.Rectangle(
        (ROI_IQ[0], ROI_IQ[2]),  # (x, y) lower left corner
        ROI_IQ[1] - ROI_IQ[0],  # width
        ROI_IQ[3] - ROI_IQ[2],  # height
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax_polar.add_patch(rect_q)

    # Draw ROI_TAU rectangle
    rect_tau = patches.Rectangle(
        (ROI_ITAU[0], ROI_ITAU[2]),  # (x, y) lower left corner
        ROI_ITAU[1] - ROI_ITAU[0],  # width
        ROI_ITAU[3] - ROI_ITAU[2],  # height
        linewidth=2,
        edgecolor="magenta",
        facecolor="none",
    )
    ax_polar.add_patch(rect_tau)

    ax_polar.set_xlabel("q (A$^{-1}$)")
    ax_polar.set_ylabel("tau (deg)")
    ax_polar.set_title(
        f"{sample_name} idx={idx}, p={pressure}[mN/m] ({suffix}) - I(q, τ) [τ = arctan(qz/qxy)]"
    )
    fig_polar.colorbar(im_polar, ax=ax_polar)

    plt.tight_layout()
    out_polar = plot_path / f"{sample_name}_{idx}_{pressure}_{suffix}_polar.png"
    fig_polar.savefig(out_polar)
    plt.close(fig_polar)


def plot_2d_maps(sample_dir, plot_path, is_water=False):
    """
    Generate 2‑D visualisations (cartesian, polar, histograms) for each frame
    stored in *cartesian*.nc files. For each cartesian file we locate the corresponding
    polar file (same prefix, ending in ``_polar.nc``), parse the index and pressure
    from the filename, and call ``_plot_2d_maps_helper``.
    """
    sample_name = sample_dir.name
    # Find cartesian files. Water samples may have a simplified naming pattern.
    cart_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_cart.nc")))
    if is_water and not cart_files:
        cart_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_cart.nc")))

    # intensity scaling – keep None for water, could be tuned for samples
    vmin = None
    vmax = None

    for cart_path in cart_files:
        # Skip background files (they're handled separately and don't have polar pairs)
        if "_bg_invquad_cart.nc" in cart_path:
            continue

        # Skip fit mask files (they are not for 2D plotting and have no polar counterpart)
        if "_fit_mask_cart.nc" in cart_path:
            continue

        # Extract index and pressure from the filename
        idx, pressure = parse_index_pressure(cart_path, is_water)

        # Determine suffix for labeling (orig / sub_water / sub_invquad) if present in the filename
        if "_orig_" in cart_path:
            suffix = "orig"
        elif "_sub_invquad_" in cart_path:
            suffix = "sub_invquad"
        elif "_sub_water_" in cart_path:
            suffix = "sub_water"
        elif "_sub_" in cart_path:
            suffix = "sub"
        else:
            suffix = ""

        # Load fit mask only for orig cart files
        mask_da = None
        if suffix == "orig":
            # Extract base name without suffix to match mask filename
            cart_path_obj = Path(cart_path)
            base_name = cart_path_obj.name.replace(f"_{suffix}_cart.nc", "")
            mask_path = cart_path_obj.parent / f"{base_name}_fit_mask_cart.nc"
            try:
                mask_da = xr.open_dataarray(mask_path)
            except FileNotFoundError:
                pass  # No mask, continue without overlay
            except Exception as e:
                print(f"Failed to load mask for {cart_path}: {e}")

        # Derive the matching polar file name (only for subtracted data)
        polar_da = None
        if suffix != "orig":
            polar_path = cart_path.replace("_cart.nc", "_polar.nc")
            try:
                polar_da = xr.open_dataarray(polar_path)
            except Exception as e:
                print(f"Failed to load polar data for {cart_path}: {e}")
                continue

        try:
            cart_da = xr.open_dataarray(cart_path)
        except Exception as e:
            print(f"Failed to load data for {cart_path}: {e}")
            continue

        _plot_2d(
            cart_da,
            polar_da,
            sample_name,
            idx,
            pressure,
            suffix,
            plot_path,
            vmin=vmin,
            vmax=vmax,
            mask_da=mask_da,
        )


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
        tau_max_data = plot_1d_profiles(sample_dir, plot_path, is_water)
        plot_2d_maps(sample_dir, plot_path, is_water)

        # Store tau_max data for this sample (skip water for tau_max plot)
        if tau_max_data and not is_water:
            all_tau_max_data[sample_name] = tau_max_data

        # Plot background files (only for samples with invquad subtraction)
        if not is_water:
            plot_background_files(sample_dir, plot_path, is_water)
            plot_horizontal_slice_comparison(sample_dir, plot_path, is_water)

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


def plot_background_files(sample_dir, plot_path, is_water=False):
    """Plot inverse quadratic background files and comparisons."""
    sample_name = sample_dir.name

    # Find all background files
    background_files = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_bg_invquad_cart.nc"))
    )

    if not background_files:
        return

    for bg_path in background_files:
        # Extract index and pressure from filename
        idx, pressure = parse_index_pressure(bg_path, is_water)

        try:
            bg_da = xr.open_dataarray(bg_path)
        except Exception as e:
            print(f"Failed to load background data for {bg_path}: {e}")
            continue

        # Plot 2D background visualization
        fig_bg = plt.figure(figsize=(8, 8))
        ax_bg = fig_bg.add_subplot(111)

        extent = (
            float(bg_da["qxy"][0]),
            float(bg_da["qxy"][-1]),
            float(bg_da["qz"][0]),
            float(bg_da["qz"][-1]),
        )

        im_bg = ax_bg.imshow(
            bg_da.values,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        ax_bg.set_xlabel("qxy (Å$^{-1}$)")
        ax_bg.set_ylabel("qz (Å$^{-1}$)")
        ax_bg.set_title(
            f"{sample_name} idx={idx}, p={pressure} mN/m\n"
            f"Fitted Background: I(qxy,qz) = (A·qz + B)·qxy$^{{-2}}$ + C"
        )
        fig_bg.colorbar(im_bg, ax=ax_bg, label="Intensity (a.u.)")
        plt.tight_layout()

        out_bg = plot_path / f"{sample_name}_{idx}_{pressure}_bg_invquad_cart.png"
        fig_bg.savefig(out_bg)
        plt.close(fig_bg)


def plot_horizontal_slice_comparison(sample_dir, plot_path, is_water=False):
    """Plot horizontal slice profiles for original data, background, and difference."""
    sample_name = sample_dir.name

    # Find all horizontal slice original files
    orig_files = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_horizontal_slice_orig.nc"))
    )

    if not orig_files:
        return

    for orig_path in orig_files:
        # Extract index and pressure from filename
        idx, pressure = parse_index_pressure(orig_path, is_water)

        try:
            # Derive paths for bg and diff
            base_path = orig_path.replace("_horizontal_slice_orig.nc", "")
            bg_path = base_path + "_horizontal_slice_bg.nc"
            diff_path = base_path + "_horizontal_slice_diff.nc"

            # Load the three DataArrays
            orig_da = xr.open_dataarray(orig_path)
            bg_da = xr.open_dataarray(bg_path)
            diff_da = xr.open_dataarray(diff_path)

            # Load fit mask and compute union over qz range
            # Format pressure as integer if it's a whole number to match file naming convention
            pressure_str = (
                str(int(pressure)) if pressure == int(pressure) else str(pressure)
            )
            mask_path = (
                sample_dir / f"{sample_name}_{idx}_{pressure_str}_fit_mask_cart.nc"
            )
            fitted_mask = None
            try:
                mask_da = xr.open_dataarray(mask_path)
                qz_min = orig_da.attrs["qz_range_min"]
                qz_max = orig_da.attrs["qz_range_max"]
                qz_slice = mask_da.sel(qz=slice(qz_min, qz_max))
                fitted_mask = np.any(qz_slice.values, axis=0)
            except Exception as e:
                print(f"Failed to load mask for horizontal slice {orig_path}: {e}")

        except Exception as e:
            print(f"Failed to load horizontal slice files for {orig_path}: {e}")
            continue

        try:
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            qxy_vals = orig_da["qxy"].values
            orig_vals = orig_da.values
            bg_vals = bg_da.values
            diff_vals = diff_da.values

            # Plot original data
            ax.plot(
                qxy_vals, orig_vals, "b-", linewidth=2, label="Original data", alpha=0.8
            )

            # Overlay fit points on original data
            if fitted_mask is not None:
                ax.scatter(
                    qxy_vals[fitted_mask],
                    orig_vals[fitted_mask],
                    color="red",
                    marker="x",
                    s=20,
                    zorder=5,
                    label="Fit points",
                )

            # Plot background
            ax.plot(
                qxy_vals,
                bg_vals,
                "r--",
                linewidth=2,
                label="Fitted background",
                alpha=0.8,
            )

            # Plot difference (residual)
            ax2 = ax.twinx()
            ax2.plot(
                qxy_vals,
                diff_vals,
                "g-",
                linewidth=1,
                label="Residual (original - background)",
                alpha=0.6,
            )
            ax2.set_ylabel("Residual Intensity (a.u.)", color="g")
            ax2.tick_params(axis="y", labelcolor="g")

            # Labels and title
            ax.set_xlabel("qxy (Å$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(
                f"{sample_name} idx={idx}, p={pressure} mN/m\n"
                f"Horizontal slice comparison (qz mean: {orig_da.attrs['qz_range_min']:.3f}-{orig_da.attrs['qz_range_max']:.3f})"
            )

            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            # Grid
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out_plot = (
                plot_path / f"{sample_name}_{idx}_{pressure}_horizontal_slice.png"
            )
            fig.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"Failed to plot horizontal slice comparison for {orig_path}: {e}")


if __name__ == "__main__":
    main()
