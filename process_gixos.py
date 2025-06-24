from pathlib import Path
import numpy as np
import pandas as pd

# User-editable constants
DATA_PATH = Path('./data/gixos')  # Path to GIXOS data directory
PROCESSED_DIR = Path('processed/gixos')
HEADER_SKIP = 30  # Number of header lines to skip in .dat files

# Dataset configurations (edit as needed)
DATASET_CONFIGS = [
    {
        'name': 'azotrans',
        'indices': [49, 53, 57, 61],
        'pressures': [0.5, 10, 20, 30],
    },
    {
        'name': 'azocis',
        'indices': [77, 81, 85, 89],
        'pressures': [5, 10, 20, 30],
    },
    {
        'name': 'azocis02',
        'indices': [105, 109],
        'pressures': [3.3, 30],
    },
    {
        'name': 'azocis03',
        'indices': [114, 118],
        'pressures': [0.1, 30],
    },
    {
        'name': 'dopc',
        'indices': [15, 10, 19, 23],
        'pressures': [0.1, 10, 20, 30],
    },
    {
        'name': 'redazo',
        'indices': [127, 131, 135, 139],
        'pressures': [0.1, 10, 20, 30],
    },
]

def process_sample(data_path, name, index, pressure, processed_dir):
    sample_dir = processed_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    dat_path = data_path / name / f"{name}_{index:05d}_SF.dat"
    if not dat_path.exists():
        print(f"Skipping: File not found {dat_path}.")
        return
    data = np.loadtxt(dat_path, skiprows=HEADER_SKIP)
    qz = data[:, 0]
    sf = data[:, 1]
    df = pd.DataFrame({'qz': qz, 'sf': sf, 'pressure': pressure, 'index': index})
    out_csv = sample_dir / f"{name}_{index}_{pressure}_sf.csv"
    df.to_csv(out_csv, index=False)
    print(f"Processed {out_csv}.")

def main():
    processed_dir = PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("Starting GIXOS processing.")
    for config in DATASET_CONFIGS:
        name = config['name']
        indices = config['indices']
        pressures = config['pressures']
        for idx, index in enumerate(indices):
            pressure = pressures[idx] if idx < len(pressures) else None
            process_sample(DATA_PATH, name, index, pressure, processed_dir)
    print("GIXOS processing completed.")

if __name__ == '__main__':
    main() 