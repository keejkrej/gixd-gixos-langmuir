#!/bin/bash

# Clean up processed and plot directories
rm -rf processed/gixd
rm -rf plot/gixd

# Run processing script
uv run process_gixd.py

# Run plotting script
uv run plot_gixd.py

echo "GIXD pipeline completed."
