from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Configure matplotlib for publication quality
plt.style.use("default")
rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "font.sans-serif": ["Arial", "DejaVu Sans", "Bitstream Vera Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "text.usetex": False,  # Set to True if LaTeX is available
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# USER SPECIFIED CONSTANTS
SAMPLE_NAME = "azocis"  # Change to your sample name
VARIABLE_NAME = "90_30_sub_invquad_cart"  # Change to your variable name
TITLE = "AzoPC-cis pressure=30.0 mN/m"  # Change to your caption
OUTPUT_FILENAME = "azocis_90_30_sub_cart.png"  # Change to your desired filename

# Additional styling options
COLORMAP = "inferno"  # Professional colormap: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
COLORBAR_LABEL = "Intensity (a.u.)"
X_LABEL = r"$q_{xy}$ [Å$^{-1}$]"  # Using proper LaTeX-style formatting
Y_LABEL = r"$q_z$ [Å$^{-1}$]"  # Using proper LaTeX-style formatting
FIGURE_SIZE = (4, 4)  # Standard single-column figure size (in inches)

# Set up paths
processed_dir = Path("processed/gixd")
sample_dir = processed_dir / SAMPLE_NAME
maps_file = sample_dir / f"{SAMPLE_NAME}_2d_maps.nc"

# Load the data
ds_maps = xr.open_dataset(maps_file)

# Print all available variable names
# print("Available variables in the dataset:")
# for var_name in ds_maps.data_vars:
#     print(f"  {var_name}")

# Get the specific variable
target_var = ds_maps[VARIABLE_NAME]

# Create the plot with publication-quality figure size
fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE, constrained_layout=True)

# Determine appropriate color scale limits
data = target_var.values
vmin = np.percentile(data[~np.isnan(data)], 1)  # 1st percentile to avoid outliers
vmax = np.percentile(data[~np.isnan(data)], 99)  # 99th percentile to avoid outliers

# Plot the data
im = ax.imshow(
    data,
    origin="lower",
    extent=(
        float(target_var["qxy"][0]),
        float(target_var["qxy"][-1]),
        float(target_var["qz"][0]),
        float(target_var["qz"][-1]),
    ),
    aspect="auto",
    cmap=COLORMAP,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",  # Better for scientific data
)

# Add colorbar with proper formatting
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(COLORBAR_LABEL, fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Set labels with proper formatting
ax.set_xlabel(X_LABEL, fontsize=10)
ax.set_ylabel(Y_LABEL, fontsize=10)

# Add subtle grid for better readability
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Set tick parameters
ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=4)
ax.tick_params(axis="both", which="minor", width=0.6, length=2)

# Add title
ax.set_title(TITLE, fontsize=12, style="italic")

# Save the figure
plot_dir = Path("plot/paper")
plot_dir.mkdir(parents=True, exist_ok=True)
save_path = plot_dir / OUTPUT_FILENAME
fig.savefig(
    save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
)
# plt.close(fig)
print(f"Saved figure to {save_path}")
print(f"Figure title: {TITLE}")

# Also create a high-resolution version for publication
high_res_path = plot_dir / OUTPUT_FILENAME.replace(".png", "_high_res.png")
fig.savefig(
    high_res_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none"
)
print(f"Saved high-resolution figure to {high_res_path}")

# Uncomment to display the figure interactively
plt.show()
