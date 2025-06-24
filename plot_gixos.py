from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob

PROCESSED_DIR = Path('processed/gixos')
PLOT_DIR = Path('plot/gixos')

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
        label = f"idx={df['index'][0]}, p={df['pressure'][0]}"
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