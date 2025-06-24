# User-editable constants
PROCESSED_DIR = 'processed/gixd'
PLOT_DIR = 'plot/gixd'  # Set to desired plot output directory

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import glob
import re

def parse_index_pressure(filename):
    # Example: azotrans_54_10_cartesian.tif -> index=54, pressure=10.0
    # Updated regex to handle decimal points directly in pressure and 'NA'
    m = re.search(r'_(\d+)_([\d.]+|NA)_', filename)
    if m:
        idx = int(m.group(1))
        pressure_str = m.group(2)
        if pressure_str == 'NA':
            pressure = 'NA'
        else:
            try:
                pressure = float(pressure_str)
            except ValueError:
                pressure = pressure_str # Keep as string if conversion fails
        return idx, pressure
    return None, None

def main():
    processed_root = Path(PROCESSED_DIR)
    plot_root = Path(PLOT_DIR)
    for sample_dir in sorted(processed_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_name = sample_dir.name
        plot_path = plot_root / sample_name
        plot_path.mkdir(parents=True, exist_ok=True)

        # Find all 1D q and theta files for the sample
        q_files = sorted(glob.glob(str(sample_dir / f'{sample_name}_*_*_intensity_q.csv')))
        theta_files = sorted(glob.glob(str(sample_dir / f'{sample_name}_*_*_intensity_theta.csv')))

        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        # Plot all intensity(q) with uniform vertical offsets of 2, ordered by pressure
        q_file_info = []
        for f in q_files:
            idx, pressure = parse_index_pressure(f)
            q_file_info.append((f, pressure, idx))
        q_file_info.sort(key=lambda x: (float('inf') if x[1] == 'NA' else x[1], x[2]))  # sort by pressure, then index
        offset_q = 0
        offset_step_q = 2
        for i, (f, pressure, idx) in enumerate(q_file_info):
            df = pd.read_csv(f)
            label = f'idx={idx}, p={pressure}'
            axs[0].plot(df['q'], df['intensity_polar'] + offset_q, label=label)
            offset_q += offset_step_q
        axs[0].set_xlabel('q (A$^{-1}$)')
        axs[0].set_ylabel('Intensity (a.u.)')
        axs[0].set_title('I(q)')
        axs[0].legend()
        axs[0].set_ylim(-5, 10 + len(q_file_info) * offset_step_q)

        # Plot all intensity(theta) with uniform vertical offsets of 2, ordered by pressure
        theta_file_info = []
        for f in theta_files:
            idx, pressure = parse_index_pressure(f)
            theta_file_info.append((f, pressure, idx))
        theta_file_info.sort(key=lambda x: (float('inf') if x[1] == 'NA' else x[1], x[2]))  # sort by pressure, then index
        offset_theta = 0
        offset_step_theta = 2
        for i, (f, pressure, idx) in enumerate(theta_file_info):
            df = pd.read_csv(f)
            label = f'idx={idx}, p={pressure}'
            axs[1].plot(df['theta'], df['intensity_polar'] + offset_theta, label=label)
            offset_theta += offset_step_theta
        axs[1].set_xlabel('theta (deg)')
        axs[1].set_ylabel('Intensity (a.u.)')
        axs[1].set_title('I(θ)')
        axs[1].legend()
        axs[1].set_ylim(-5, 10 + len(theta_file_info) * offset_step_theta)

        fig.suptitle(sample_name)
        plt.tight_layout()
        out1d = plot_path / f'{sample_name}_1d_profiles.png'
        fig.savefig(out1d)
        plt.close(fig)
        print(f"Saved 1D profiles to {out1d}")

        # Locate all Cartesian TIFF files for the chosen sample
        cart_files = sorted(
            glob.glob(str(sample_dir / f"{sample_name}_*_*_cartesian.tif"))
        )

        for cart_path in cart_files:
            idx, pressure = parse_index_pressure(cart_path)

            # Corresponding polar data paths
            polar_path = cart_path.replace("_cartesian.tif", "_polar.tif")
            # New: axis CSVs are now separate files
            cart_qxy_csv = cart_path.replace("_cartesian.tif", "_cartesian_qxy.csv")
            cart_qz_csv = cart_path.replace("_cartesian.tif", "_cartesian_qz.csv")
            polar_q_csv = polar_path.replace("_polar.tif", "_polar_q.csv")
            polar_theta_csv = polar_path.replace("_polar.tif", "_polar_theta.csv")

            # Ensure polar file exists
            if not Path(polar_path).exists():
                print(f"Skipping {cart_path}: polar file not found.")
                continue

            # Load images
            cart_img = tifffile.imread(cart_path)
            polar_img = tifffile.imread(polar_path)

            # Load axis arrays for extent
            try:
                qxy = pd.read_csv(cart_qxy_csv)['qxy'].values
                qz = pd.read_csv(cart_qz_csv)['qz'].values
                q = pd.read_csv(polar_q_csv)['q'].values
                theta = pd.read_csv(polar_theta_csv)['theta'].values
            except Exception as e:
                print(f"Warning: Could not load axes for {cart_path}. Skipping 2D map.")
                continue

            # Create figure for this scan
            fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

            # Cartesian map with uniform contrast
            im0 = axs2[0].imshow(
                cart_img,
                origin="lower",
                extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
                aspect="auto",
                vmin=0,
                vmax=10
            )
            axs2[0].set_xlabel("qxy (A$^{-1}$)")
            axs2[0].set_ylabel("qz (A$^{-1}$)")
            axs2[0].set_title(
                "I(qxy, qz)",
                loc='center'
            )
            fig2.colorbar(im0, ax=axs2[0])

            # Polar map with uniform contrast
            im1 = axs2[1].imshow(
                polar_img,
                origin="lower",
                extent=[q[0], q[-1], theta[0], theta[-1]],
                aspect="auto",
                vmin=0,
                vmax=10
            )
            axs2[1].set_xlabel("q (A$^{-1}$)")
            axs2[1].set_ylabel("theta (deg)")
            axs2[1].set_title(
                "I(q, θ)",
                loc='center'
            )
            fig2.colorbar(im1, ax=axs2[1])

            fig2.suptitle(f"{sample_name} idx={idx}, p={pressure}")
            out2d = plot_path / f'{sample_name}_{idx}_{pressure}_2d_maps.png'
            fig2.savefig(out2d)
            plt.close(fig2)
            print(f"Saved 2D maps to {out2d}")

if __name__ == '__main__':
    main() 