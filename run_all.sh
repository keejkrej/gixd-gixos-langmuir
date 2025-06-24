#!/bin/bash
set -e

python3 process_gixd.py
python3 process_gixos.py

python3 plot_gixd.py
python3 plot_gixos.py