# User-editable constants
DATA_PATH = './data/gixd'  # Path to gixd data directory
# Water reference (name, index) or None
WATER_REF = ('water', 44)
# Where to save all processed results
PROCESSED_DIR = 'processed/gixd'
# ROI for 1D extraction (as in gixd.py)
ROI_Q = [0.7, 2.0, 0, 10]        # [q_min, q_max, theta_min, theta_max]
ROI_THETA = [1.25, 1.5, 0, 60]   # [q_min, q_max, theta_min, theta_max]

# Dataset configurations (name, indices, pressures)
DATASET_CONFIGS = [
    {'name': 'azotrans', 'indices': [54, 58, 62], 'pressures': [10, 20, 30]},
    {'name': 'azocis', 'indices': [78, 82, 86, 90], 'pressures': [5, 10, 20, 30]},
    {'name': 'azocis02', 'indices': [106, 110], 'pressures': [3.3, 30]},
    {'name': 'azocis03', 'indices': [115, 119], 'pressures': [0.1, 30]},
    {'name': 'dopc', 'indices': [16, 12, 20, 24], 'pressures': [0.1, 10, 20, 30]},
    {'name': 'redazo', 'indices': [128, 132, 136, 140], 'pressures': [0.1, 10, 20, 30]},
]

from pathlib import Path
import tifffile
import pandas as pd
import numpy as np
from utils.data.gixd import load_gixd_xarray, gixd_cartesian2polar, extract_intensity_q, extract_intensity_theta

def process_sample(data_path, name, index, pressure, processed_dir, da_water=None):
    processed_dir = Path(processed_dir) / name
    processed_dir.mkdir(parents=True, exist_ok=True)
    da_cart = load_gixd_xarray(data_path, name, index)
    if da_water is not None:
        if da_cart.shape == da_water.shape:
            da_cart = da_cart - da_water
            print("Water reference subtracted.")
        else:
            print(f"Water ref mismatch for {name}_{index}. Skipping subtraction.")
    # Format pressure for filename
    pressure_str = str(pressure) if pressure is not None else 'NA'
    # Save intensity(qxy, qz) as TIFF
    tifffile.imwrite(str(processed_dir / f'{name}_{index}_{pressure_str}_cartesian.tif'), da_cart.values)
    # Save axes as separate one-column CSV files
    pd.DataFrame({'qxy': da_cart['qxy'].values}).to_csv(
        processed_dir / f'{name}_{index}_{pressure_str}_cartesian_qxy.csv', index=False)
    pd.DataFrame({'qz': da_cart['qz'].values}).to_csv(
        processed_dir / f'{name}_{index}_{pressure_str}_cartesian_qz.csv', index=False)
    # Convert to polar
    da_polar = gixd_cartesian2polar(da_cart)
    tifffile.imwrite(str(processed_dir / f'{name}_{index}_{pressure_str}_polar.tif'), da_polar.values)
    pd.DataFrame({'q': da_polar['q'].values}).to_csv(
        processed_dir / f'{name}_{index}_{pressure_str}_polar_q.csv', index=False)
    pd.DataFrame({'theta': da_polar['theta'].values}).to_csv(
        processed_dir / f'{name}_{index}_{pressure_str}_polar_theta.csv', index=False)
    # 1D profiles using ROIs
    intensity_q = extract_intensity_q(
        da_polar,
        q_range=(ROI_Q[0], ROI_Q[1]),
        theta_range=(ROI_Q[2], ROI_Q[3])
    )
    df_q = intensity_q.to_dataframe().reset_index()
    df_q.to_csv(processed_dir / f'{name}_{index}_{pressure_str}_intensity_q.csv', index=False)
    intensity_theta = extract_intensity_theta(
        da_polar,
        q_range=(ROI_THETA[0], ROI_THETA[1]),
        theta_range=(ROI_THETA[2], ROI_THETA[3])
    )
    df_theta = intensity_theta.to_dataframe().reset_index()
    df_theta.to_csv(processed_dir / f'{name}_{index}_{pressure_str}_intensity_theta.csv', index=False)
    print(f"Processed data for {name}_{index}_{pressure_str}.")

def main():
    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Load water reference if specified
    da_water = None
    if WATER_REF is not None:
        water_name, water_index = WATER_REF
        da_water = load_gixd_xarray(DATA_PATH, water_name, water_index)
        print(f"Loaded water reference: {water_name}_{water_index}.")
    # Process all datasets
    print("Starting GIXD processing.")
    for config in DATASET_CONFIGS:
        name = config['name']
        indices = config['indices']
        pressures = config['pressures']
        for idx, index in enumerate(indices):
            pressure = pressures[idx] if idx < len(pressures) else None
            process_sample(DATA_PATH, name, index, pressure, processed_dir, da_water)
    print("GIXD processing completed.")

if __name__ == '__main__':
    main() 