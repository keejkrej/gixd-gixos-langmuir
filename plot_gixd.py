from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob
import re

# from scipy.ndimage import median_filter
# from utils.math.com import exponential_weighted_center
from data_gixd import ROI_IQ, ROI_ITAU


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
    pressure = float(m.group(2))
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
    """Shared helper to plot 1‑D intensity profiles.

    Parameters
    ----------
    file_info: list of tuples ``(filepath, pressure, idx)``
    axis_name: ``"q"`` or ``"tau"`` – the coordinate dimension in the DataArray.
    xlabel, ylabel, title: Plot labels.
    out_path: Destination path for the saved PNG.
    tau_max_data: Dictionary that will be populated with ``pressure -> argmax`` of the
        intensity for each file (mirrors the original implementation).
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for i, (f, pressure, idx) in enumerate(file_info):
        da = xr.open_dataarray(f)
        intensity = da.values
        axis_vals = da[axis_name].values

        # Label suffix distinguishes original vs water‑subtracted data
        suffix = "orig" if "_orig_" in f else ("sub" if "_sub_" in f else "")
        label = f"idx={idx}, p={pressure}[mN/m]" + (f" ({suffix})" if suffix else "")

        ax.plot(
            axis_vals,
            intensity,
            "k-",
            linewidth=2,
            label=f"{label} (raw)",
        )
        # Record the index of the maximum intensity for later tau‑max analysis.
        tau_max_data[pressure] = np.argmax(intensity)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_1d_profiles(sample_dir, plot_path, is_water=False):
    sample_name = sample_dir.name

    q_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_intensity_q.nc")))
    tau_files = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_intensity_tau.nc"))
    )

    # Handle water case (single file)
    if is_water and not q_files:
        q_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_intensity_q.nc")))
    if is_water and not tau_files:
        tau_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_intensity_tau.nc"))
        )

    # Dictionary to store tau_max vs pressure for this sample
    tau_max_data = {}

    # Plot I(q) profiles
    if q_files:
        q_file_info = []
        for f in q_files:
            idx, pressure = parse_index_pressure(f, is_water)
            q_file_info.append((f, pressure, idx))
        q_file_info.sort(key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2]))

        out_q = plot_path / f"{sample_name}_Iq_profiles.png"
        _plot_1d(
            q_file_info,
            axis_name="q",
            xlabel="q (A$^{-1}$)",
            ylabel="Intensity (a.u.)",
            title=f"{sample_name} - I(q)",
            out_path=out_q,
            tau_max_data=tau_max_data,
        )

    # Plot I(tau) profiles using the shared helper
    if tau_files:
        tau_file_info = []
        for f in tau_files:
            idx, pressure = parse_index_pressure(f, is_water)
            tau_file_info.append((f, pressure, idx))
        tau_file_info.sort(key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2]))

        out_tau = plot_path / f"{sample_name}_Itau_profiles.png"
        _plot_1d(
            tau_file_info,
            axis_name="tau",
            xlabel="tau (deg)",
            ylabel="Intensity (a.u.)",
            title=f"{sample_name} - I(τ)",
            out_path=out_tau,
            tau_max_data=tau_max_data,
        )

    return tau_max_data


def _plot_2d(
    cart_da: xr.DataArray,
    polar_da: xr.DataArray,
    sample_name: str,
    idx: int,
    pressure: float,
    suffix: str,
    plot_path: Path,
    vmax: float | None = None,
):
    """Helper to generate the three 2‑D visualisations for a single frame."""
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
        vmin=0,
        vmax=vmax,
    )
    ax_cart.set_xlabel("qxy (A$^{-1}$)")
    ax_cart.set_ylabel("qz (A$^{-1}$)")
    ax_cart.set_title(
        f"{sample_name} idx={idx}, p={pressure}[mN/m] ({suffix}) - I(qxy, qz)"
    )
    fig_cart.colorbar(im_cart, ax=ax_cart)

    out_cart = plot_path / f"{sample_name}_{idx}_{pressure}_{suffix}_Iqxyqz.png"
    fig_cart.savefig(out_cart)
    plt.close(fig_cart)

    # Plot intensity histogram for cartesian map
    fig_hist_cart = plt.figure(figsize=(6, 4))
    ax_hist_cart = fig_hist_cart.add_subplot(111)
    ax_hist_cart.hist(
        cart_da.values.ravel(), bins=50, color="steelblue", edgecolor="black"
    )
    ax_hist_cart.set_xlabel("Intensity (a.u.)")
    ax_hist_cart.set_ylabel("Frequency")
    ax_hist_cart.set_title(
        f"{sample_name} idx={idx}, p={pressure}[mN/m] - Intensity Histogram (Cartesian)"
    )
    out_hist_cart = (
        plot_path / f"{sample_name}_{idx}_{pressure}_{suffix}_Iqxyqz_hist.png"
    )
    fig_hist_cart.savefig(out_hist_cart)
    plt.close(fig_hist_cart)

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
        vmin=0,
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

    out_polar = plot_path / f"{sample_name}_{idx}_{pressure}_{suffix}_Iqtau.png"
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
    cart_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_cartesian.nc")))
    if is_water and not cart_files:
        cart_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_cartesian.nc"))
        )

    # intensity scaling – keep None for water, could be tuned for samples
    vmax = None

    for cart_path in cart_files:
        # Extract index and pressure from the filename
        idx, pressure = parse_index_pressure(cart_path, is_water)

        # Determine suffix for labeling (orig / sub) if present in the filename
        suffix = (
            "orig" if "_orig_" in cart_path else ("sub" if "_sub_" in cart_path else "")
        )

        # Derive the matching polar file name
        polar_path = cart_path.replace("_cartesian.nc", "_polar.nc")

        try:
            cart_da = xr.open_dataarray(cart_path)
            polar_da = xr.open_dataarray(polar_path)
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
            vmax=vmax,
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

        ax_tau_max.set_xlabel("Pressure")
        ax_tau_max.set_ylabel("Tau_max (deg)")
        ax_tau_max.set_title("Tau_max vs Pressure")
        ax_tau_max.legend()
        ax_tau_max.grid(True, alpha=0.3)

        plt.tight_layout()
        out_tau_max = plot_dir / "tau_max_vs_pressure.png"
        fig_tau_max.savefig(out_tau_max)
        plt.close(fig_tau_max)

    print("GIXD plotting completed.")


if __name__ == "__main__":
    main()
