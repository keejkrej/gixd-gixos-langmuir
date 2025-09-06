from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import glob
import re


PROCESSED_DIR = "processed/gixd"
PLOT_DIR = "plot/gixd"


def parse_index_pressure(filename):
    m = re.search(r"_(\d+)_([\d.]+|NA)_", filename)
    if not m:
        raise ValueError(f"Could not parse index and pressure from filename: {filename}")
    idx = int(m.group(1))
    pressure = float(m.group(2))
    return idx, pressure


def main():
    processed_dir = Path(PROCESSED_DIR)
    plot_dir = Path(PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir in sorted(processed_dir.iterdir()):
        sample_name = sample_dir.name
        plot_path = plot_dir / sample_name
        plot_path.mkdir(parents=True, exist_ok=True)

        cart_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_*_cartesian.nc"))
        )
        polar_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_*_polar.nc")))
        q_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_*_intensity_q.nc"))
        )
        theta_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_*_intensity_theta.nc"))
        )

        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        # Plot all intensity(q) with uniform vertical offsets of 2, ordered by pressure
        q_file_info = []
        for f in q_files:
            idx, pressure = parse_index_pressure(f)
            q_file_info.append((f, pressure, idx))
        q_file_info.sort(
            key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2])
        )  # sort by pressure, then index
        offset_q = 0
        offset_step_q = 2
        for i, (f, pressure, idx) in enumerate(q_file_info):
            da = xr.open_dataarray(f)
            label = f"idx={idx}, p={pressure}"
            axs[0].plot(da["q"], da.values + offset_q, label=label)
            offset_q += offset_step_q
        axs[0].set_xlabel("q (A$^{-1}$)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].set_title("I(q)")
        axs[0].legend()
        axs[0].set_ylim(-5, 10 + len(q_file_info) * offset_step_q)

        # Plot all intensity(theta) with uniform vertical offsets of 2, ordered by pressure
        theta_file_info = []
        for f in theta_files:
            idx, pressure = parse_index_pressure(f)
            theta_file_info.append((f, pressure, idx))
        theta_file_info.sort(
            key=lambda x: (float("inf") if x[1] == "NA" else x[1], x[2])
        )  # sort by pressure, then index
        offset_theta = 0
        offset_step_theta = 2
        for i, (f, pressure, idx) in enumerate(theta_file_info):
            da = xr.open_dataarray(f)
            label = f"idx={idx}, p={pressure}"
            axs[1].plot(da["theta"], da.values + offset_theta, label=label)
            offset_theta += offset_step_theta
        axs[1].set_xlabel("theta (deg)")
        axs[1].set_ylabel("Intensity (a.u.)")
        axs[1].set_title("I(θ)")
        axs[1].legend()
        axs[1].set_ylim(-5, 10 + len(theta_file_info) * offset_step_theta)

        fig.suptitle(sample_name)
        plt.tight_layout()
        out1d = plot_path / f"{sample_name}_1d_profiles.png"
        fig.savefig(out1d)
        plt.close(fig)
        print(f"Saved 1D profiles to {out1d}")

        # Locate all Cartesian NetCDF files for the chosen sample

        for cart_path in cart_files:
            idx, pressure = parse_index_pressure(cart_path)

            # Corresponding polar data paths
            polar_path = cart_path.replace("_cartesian.nc", "_polar.nc")

            # Ensure polar file exists
            if not Path(polar_path).exists():
                print(f"Skipping {cart_path}: polar file not found.")
                continue

            # Load NetCDF data
            try:
                cart_da = xr.open_dataarray(cart_path)
                polar_da = xr.open_dataarray(polar_path)
            except Exception as e:
                print(
                    f"Warning: Could not load NetCDF data for {cart_path}. Skipping 2D map."
                )
                continue

            # Create figure for this scan
            fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

            # Cartesian map with uniform contrast
            im0 = axs2[0].imshow(
                cart_da.values,
                origin="lower",
                extent=[
                    cart_da["qxy"][0],
                    cart_da["qxy"][-1],
                    cart_da["qz"][0],
                    cart_da["qz"][-1],
                ],
                aspect="auto",
                vmin=0,
                vmax=10,
            )
            axs2[0].set_xlabel("qxy (A$^{-1}$)")
            axs2[0].set_ylabel("qz (A$^{-1}$)")
            axs2[0].set_title("I(qxy, qz)", loc="center")
            fig2.colorbar(im0, ax=axs2[0])

            # Polar map with uniform contrast
            im1 = axs2[1].imshow(
                polar_da.values,
                origin="lower",
                extent=[
                    polar_da["q"][0],
                    polar_da["q"][-1],
                    polar_da["theta"][0],
                    polar_da["theta"][-1],
                ],
                aspect="auto",
                vmin=0,
                vmax=10,
            )
            axs2[1].set_xlabel("q (A$^{-1}$)")
            axs2[1].set_ylabel("theta (deg)")
            axs2[1].set_title("I(q, θ)", loc="center")
            fig2.colorbar(im1, ax=axs2[1])

            fig2.suptitle(f"{sample_name} idx={idx}, p={pressure}")
            out2d = plot_path / f"{sample_name}_{idx}_{pressure}_2d_maps.png"
            fig2.savefig(out2d)
            plt.close(fig2)
            print(f"Saved 2D maps to {out2d}")


if __name__ == "__main__":
    main()
