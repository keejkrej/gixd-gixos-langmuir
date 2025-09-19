from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob
import re
from scipy.ndimage import median_filter
from utils.math.com import exponential_weighted_center
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
        fig_q = plt.figure(figsize=(8, 8))
        ax_q = fig_q.add_subplot(111)

        q_file_info = []
        for f in q_files:
            idx, pressure = parse_index_pressure(f, is_water)
            q_file_info.append((f, pressure, idx))
        q_file_info.sort(key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2]))

        for i, (f, pressure, idx) in enumerate(q_file_info):
            da = xr.open_dataarray(f)
            intensity = da.values
            q_values = da["q"].values

            # Apply median filter for smoothing
            smoothed_intensity = median_filter(intensity, size=21)

            label = f"idx={idx}, p={pressure}[mN/m]"

            # Plot raw data
            ax_q.plot(
                q_values,
                intensity,
                "k-",
                alpha=0.5,
                linewidth=0.5,
                label=f"{label} (raw)",
            )
            # Plot smoothed data
            ax_q.plot(
                q_values,
                smoothed_intensity,
                f"C{i}",
                linewidth=2,
                label=f"{label} (smoothed)",
            )

        ax_q.set_xlabel("q (A$^{-1}$)")
        ax_q.set_ylabel("Intensity (a.u.)")
        ax_q.set_title(f"{sample_name} - I(q)")
        ax_q.legend()

        plt.tight_layout()
        out_q = plot_path / f"{sample_name}_Iq_profiles.png"
        fig_q.savefig(out_q)
        plt.close(fig_q)

    # Plot I(tau) profiles
    if tau_files:
        fig_tau = plt.figure(figsize=(8, 8))
        ax_tau = fig_tau.add_subplot(111)

        tau_file_info = []
        for f in tau_files:
            idx, pressure = parse_index_pressure(f, is_water)
            tau_file_info.append((f, pressure, idx))
        tau_file_info.sort(key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2]))

        for i, (f, pressure, idx) in enumerate(tau_file_info):
            da = xr.open_dataarray(f)
            intensity = da.values
            tau_values = da["tau"].values

            # Apply median filter for smoothing
            smoothed_intensity = median_filter(intensity, size=21)

            label = f"idx={idx}, p={pressure}[mN/m]"

            # Plot raw data
            ax_tau.plot(
                tau_values,
                intensity,
                "k-",
                alpha=0.5,
                linewidth=0.5,
                label=f"{label} (raw)",
            )
            # Plot smoothed data
            ax_tau.plot(
                tau_values,
                smoothed_intensity,
                f"C{i}",
                linewidth=2,
                label=f"{label} (smoothed)",
            )

            # Calculate exponentially weighted center on smoothed median filter results
            if len(smoothed_intensity) > 0:
                tau_center = exponential_weighted_center(
                    smoothed_intensity, tau_values, temperature=0.1
                )

                ax_tau.plot(
                    tau_center,
                    smoothed_intensity[np.argmin(np.abs(tau_values - tau_center))],
                    "rx",
                    markersize=20,
                    markeredgewidth=4,
                )
                # Store tau_center vs pressure
                tau_max_data[pressure] = tau_center

        ax_tau.set_xlabel("tau (deg)")
        ax_tau.set_ylabel("Intensity (a.u.)")
        ax_tau.set_title(f"{sample_name} - I(τ)")
        ax_tau.legend()

        plt.tight_layout()
        out_tau = plot_path / f"{sample_name}_Itau_profiles.png"
        fig_tau.savefig(out_tau)
        plt.close(fig_tau)

    return tau_max_data


def plot_2d_maps(sample_dir, plot_path, is_water=False):
    sample_name = sample_dir.name
    cart_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_cartesian.nc")))

    vmax = None if is_water else 10
    # Handle water case (single file)
    if is_water and not cart_files:
        cart_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_cartesian.nc"))
        )

    for cart_path in cart_files:
        idx, pressure = parse_index_pressure(cart_path, is_water)
        polar_path = cart_path.replace("_cartesian.nc", "_polar.nc")

        if not Path(polar_path).exists():
            continue

        try:
            cart_da = xr.open_dataarray(cart_path)
            polar_da = xr.open_dataarray(polar_path)
        except Exception:
            continue

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
        ax_cart.set_title(f"{sample_name} idx={idx}, p={pressure}[mN/m] - I(qxy, qz)")
        fig_cart.colorbar(im_cart, ax=ax_cart)

        out_cart = plot_path / f"{sample_name}_{idx}_{pressure}_Iqxyqz.png"
        fig_cart.savefig(out_cart)
        plt.close(fig_cart)

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
            f"{sample_name} idx={idx}, p={pressure}[mN/m] - I(q, τ) [τ = arctan(qz/qxy)]"
        )
        fig_polar.colorbar(im_polar, ax=ax_polar)

        out_polar = plot_path / f"{sample_name}_{idx}_{pressure}_Iqtau.png"
        fig_polar.savefig(out_polar)
        plt.close(fig_polar)


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


if __name__ == "__main__":
    main()
