from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from scipy.ndimage import median_filter
from utils.math.com import exponential_weighted_center


PROCESSED_DIR = "processed/gixd"
PLOT_DIR = "plot/gixd"


def parse_index_pressure(filename):
    m = re.search(r"_(\d+)_([\d.]+|NA)_", filename)
    if not m:
        raise ValueError(
            f"Could not parse index and pressure from filename: {filename}"
        )
    idx = int(m.group(1))
    pressure = float(m.group(2))
    return idx, pressure


def plot_1d_profiles(sample_dir, plot_path):
    sample_name = sample_dir.name

    q_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_intensity_q.nc")))
    theta_files = sorted(
        glob.glob(str(sample_dir / f"{sample_name}_*_*_intensity_theta.nc"))
    )

    # Dictionary to store theta_max vs pressure for this sample
    theta_max_data = {}

    # Plot I(q) profiles
    if q_files:
        fig_q = plt.figure(figsize=(8, 8))
        ax_q = fig_q.add_subplot(111)

        q_file_info = []
        for f in q_files:
            idx, pressure = parse_index_pressure(f)
            q_file_info.append((f, pressure, idx))
        q_file_info.sort(key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2]))

        for i, (f, pressure, idx) in enumerate(q_file_info):
            da = xr.open_dataarray(f)
            intensity = da.values
            q_values = da["q"].values

            # Apply median filter for smoothing
            smoothed_intensity = median_filter(intensity, size=21)

            label = f"idx={idx}, p={pressure}"

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

    # Plot I(theta) profiles
    if theta_files:
        fig_theta = plt.figure(figsize=(8, 8))
        ax_theta = fig_theta.add_subplot(111)

        theta_file_info = []
        for f in theta_files:
            idx, pressure = parse_index_pressure(f)
            theta_file_info.append((f, pressure, idx))
        theta_file_info.sort(
            key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2])
        )

        for i, (f, pressure, idx) in enumerate(theta_file_info):
            da = xr.open_dataarray(f)
            intensity = da.values
            theta_values = da["theta"].values

            # Apply median filter for smoothing
            smoothed_intensity = median_filter(intensity, size=21)

            label = f"idx={idx}, p={pressure}"

            # Plot raw data
            ax_theta.plot(
                theta_values,
                intensity,
                "k-",
                alpha=0.5,
                linewidth=0.5,
                label=f"{label} (raw)",
            )
            # Plot smoothed data
            ax_theta.plot(
                theta_values,
                smoothed_intensity,
                f"C{i}",
                linewidth=2,
                label=f"{label} (smoothed)",
            )

            # Calculate exponentially weighted center on smoothed median filter results
            if len(smoothed_intensity) > 0:
                theta_center = exponential_weighted_center(
                    smoothed_intensity, theta_values, temperature=0.1
                )

                ax_theta.plot(
                    theta_center,
                    smoothed_intensity[np.argmin(np.abs(theta_values - theta_center))],
                    "rx",
                    markersize=20,
                    markeredgewidth=4,
                )
                # Store theta_center vs pressure
                theta_max_data[pressure] = theta_center

        ax_theta.set_xlabel("theta (deg)")
        ax_theta.set_ylabel("Intensity (a.u.)")
        ax_theta.set_title(f"{sample_name} - I(θ)")
        ax_theta.legend()

        plt.tight_layout()
        out_theta = plot_path / f"{sample_name}_Itheta_profiles.png"
        fig_theta.savefig(out_theta)
        plt.close(fig_theta)

    return theta_max_data


def plot_2d_maps(sample_dir, plot_path):
    sample_name = sample_dir.name
    cart_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_cartesian.nc")))

    for cart_path in cart_files:
        idx, pressure = parse_index_pressure(cart_path)
        polar_path = cart_path.replace("_cartesian.nc", "_polar.nc")

        if not Path(polar_path).exists():
            continue

        try:
            cart_da = xr.open_dataarray(cart_path)
            polar_da = xr.open_dataarray(polar_path)
        except Exception as e:
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
            vmax=10,
        )
        ax_cart.set_xlabel("qxy (A$^{-1}$)")
        ax_cart.set_ylabel("qz (A$^{-1}$)")
        ax_cart.set_title(f"{sample_name} idx={idx}, p={pressure} - I(qxy, qz)")
        fig_cart.colorbar(im_cart, ax=ax_cart)

        out_cart = plot_path / f"{sample_name}_{idx}_{pressure}_Iqxyqz.png"
        fig_cart.savefig(out_cart)
        plt.close(fig_cart)

        # Plot I(q, θ) map
        fig_polar = plt.figure(figsize=(8, 8))
        ax_polar = fig_polar.add_subplot(111)

        im_polar = ax_polar.imshow(
            polar_da.values,
            origin="lower",
            extent=(
                float(polar_da["q"][0]),
                float(polar_da["q"][-1]),
                float(polar_da["theta"][0]),
                float(polar_da["theta"][-1]),
            ),
            aspect="auto",
            vmin=0,
            vmax=10,
        )
        ax_polar.set_xlabel("q (A$^{-1}$)")
        ax_polar.set_ylabel("theta (deg)")
        ax_polar.set_title(f"{sample_name} idx={idx}, p={pressure} - I(q, θ)")
        fig_polar.colorbar(im_polar, ax=ax_polar)

        out_polar = plot_path / f"{sample_name}_{idx}_{pressure}_Iqtheta.png"
        fig_polar.savefig(out_polar)
        plt.close(fig_polar)


def main():
    processed_dir = Path(PROCESSED_DIR)
    plot_dir = Path(PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to collect theta_max vs pressure for all samples
    all_theta_max_data = {}

    for sample_dir in sorted(processed_dir.iterdir()):
        sample_name = sample_dir.name
        plot_path = plot_dir / sample_name
        plot_path.mkdir(parents=True, exist_ok=True)

        print(f"Plotting {sample_name}...")
        theta_max_data = plot_1d_profiles(sample_dir, plot_path)
        plot_2d_maps(sample_dir, plot_path)

        # Store theta_max data for this sample
        if theta_max_data:
            all_theta_max_data[sample_name] = theta_max_data

    # Plot theta_max vs pressure for all samples
    if all_theta_max_data:
        fig_theta_max = plt.figure(figsize=(8, 8))
        ax_theta_max = fig_theta_max.add_subplot(111)

        for sample_name, theta_data in all_theta_max_data.items():
            # Filter out pressures < 5 only for dopc and redazo samples
            # if sample_name.lower() in ["dopc", "redazo"]:
            #     filtered_data = {p: theta_data[p] for p in theta_data.keys() if p >= 5}
            # else:
            #     filtered_data = theta_data
            # pressures = sorted(filtered_data.keys())
            # theta_max_values = [filtered_data[p] for p in pressures]
            pressures = sorted(theta_data.keys())
            theta_max_values = [theta_data[p] for p in pressures]

            if pressures:  # Only plot if there's data left after filtering
                ax_theta_max.plot(pressures, theta_max_values, "o-", label=sample_name)

        ax_theta_max.set_xlabel("Pressure")
        ax_theta_max.set_ylabel("Theta_max (deg)")
        ax_theta_max.set_title("Theta_max vs Pressure")
        ax_theta_max.legend()
        ax_theta_max.grid(True, alpha=0.3)

        plt.tight_layout()
        out_theta_max = plot_dir / "theta_max_vs_pressure.png"
        fig_theta_max.savefig(out_theta_max)
        plt.close(fig_theta_max)


if __name__ == "__main__":
    main()
