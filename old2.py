from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

PROCESSED_DIR = Path('processed/gixos')
PLOT_DIR = Path('plot/gixos')

def parse_index_pressure(filename):
    # Example: azotrans_54_10_sf.csv -> index=54, pressure=10.0
    m = re.search(r'_(\d+)_([\d.]+|NA)_sf\.csv', filename)
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

def plot_sample(sample_name, sample_dir, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(glob.glob(str(sample_dir / f"{sample_name}_*_sf.csv")))
    if not csv_files:
        print(f"No CSVs found. Skipping plotting for this sample.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    offset = 0
    offset_step = 50
    n_curves = len(csv_files)
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        idx, pressure = parse_index_pressure(Path(csv_file).name)
        if idx is not None:
            label = f"idx={idx}, p={pressure}"
        else:
            label = Path(csv_file).name
        ax.scatter(df['qz'], df['sf'] + offset, label=label, facecolors='none', edgecolors='C'+str(i%10), alpha=1.0)
        offset += offset_step
    ax.set_xlabel('qz (A$^{-1}$)')
    ax.set_ylabel('SF (a.u.)')
    ax.set_title(f'Structure Factors for {sample_name}')
    ax.legend()
    ax.set_ylim(-50, 150 + n_curves * offset_step)
    out_png = plot_dir / f"{sample_name}_sf.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved plot to {out_png}")

def main():
    processed_root = PROCESSED_DIR
    plot_root = PLOT_DIR
    for sample_dir in sorted(processed_root.iterdir()):
        if not sample_dir.is_dir():
            print(f"Skipping non-directory item: {sample_dir.name}")
            continue
        sample_name = sample_dir.name
        plot_sample(sample_name, sample_dir, plot_root / sample_name)

if __name__ == '__main__':
    main() 