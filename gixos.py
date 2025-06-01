from pathlib import Path
from utils.data.gixos import GIXOSData, GIXOSDataset, GIXOSProject

# Define data and plot paths
DATA_PATH = Path('~/workspace/data/lipid/GIXOS').expanduser()
PLOT_PATH = Path('~/workspace/plot/lipid/GIXOS').expanduser()

# Define dataset configurations
DATASET_CONFIGS = [
    {
        'name': 'azotrans',
        'indices': [49, 53, 57, 61],
        'pressures': [0.5, 10, 20, 30],
        'peak_1': [(0.05, 0.3), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2)],
        'peak_2': [(0.3, 0.5), (0.2, 0.4), (0.2, 0.35), (0.2, 0.35)],
    },
    {
        'name': 'azocis',
        'indices': [77, 81, 85, 89],
        'pressures': [5, 10, 20, 30],
        'peak_1': [(0.05, 0.3), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2)],
        'peak_2': [(0.3, 0.5), (0.3, 0.5), (0.2, 0.4), (0.2, 0.35)],
    },
    {
        'name': 'azocis02',
        'indices': [105, 109],
        'pressures': [3.3, 30],
        'peak_1': [(0.05, 0.3), (0.05, 0.2)],
        'peak_2': [(0.3, 0.5), (0.2, 0.35)],
    },
    {
        'name': 'azocis03',
        'indices': [114, 118],
        'pressures': [0.1, 30],
        'peak_1': [(0.05, 0.3), (0.05, 0.2)],
        'peak_2': [(0.3, 0.5), (0.2, 0.35)],
    },
    {
        'name': 'dopc',
        'indices': [15, 10, 19, 23],
        'pressures': [0.1, 10, 20, 30],
        'peak_1': [(0.05, 0.3), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2)],
        'peak_2': [(0.3, 0.5), (0.2, 0.4), (0.2, 0.35), (0.2, 0.35)],
    },
    {
        'name': 'redazo',
        'indices': [127, 131, 135, 139],
        'pressures': [0.1, 10, 20, 30],
        'peak_1': [(0.05, 0.3), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2)],
        'peak_2': [(0.3, 0.5), (0.2, 0.4), (0.2, 0.35), (0.2, 0.35)],
    },
]

# Load all datasets and subtract water
datasets = {}
for config in DATASET_CONFIGS:
    dataset = GIXOSDataset.load(
        DATA_PATH,
        config['name'],
        config['indices'],
        config['pressures'],
        config['peak_1'],
        config['peak_2'],
    )
    datasets[config['name']] = dataset

project = GIXOSProject(
    data_path=DATA_PATH,
    plot_path=PLOT_PATH,
    datasets=datasets
)
project.plot_all()