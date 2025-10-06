# Paper Plotting Instructions

## Using plot_paper.py

The `plot_paper.py` script is designed to create individual figures for your paper with minimal complexity. It plots a single sub_invquad_cart 2D map without any additional features.

### How to Use

1. Open `plot_paper.py` and modify the constants at the top of the file:

   - `SAMPLE_NAME`: Change to your sample name
   - `VARIABLE_NAME`: Change to your specific variable name
   - `TITLE`: Change to your title
   - `OUTPUT_FILENAME`: Change to your desired filename

2. Run the script:

   ```bash
   python plot_paper.py
   ```

3. The figure will be saved to `plot/paper/` directory.

### Customization Options

- Figure size: Modify `figsize=(8, 6)` on line 21
- Resolution: Modify `dpi=300` on line 54
- Colormap: Modify `cmap="viridis"` on line 28
- Labels and title: Modify lines 41-45 and 49

### Creating Different Plots

To create different plots:

1. Make a copy of `plot_paper.py` with a new name (e.g., `plot_other.py`)
2. Modify the constants at the top of the file to point to a different data variable
3. Adjust other parameters as needed
4. Run the new script

This approach keeps each plotting script simple and focused on a single figure.
