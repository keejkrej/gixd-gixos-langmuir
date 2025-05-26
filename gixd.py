from pathlib import Path
from xrd_anlyutils.data.gixd import GIXDData, GIXDDataset, GIXDProject

# Define data and plot paths
DATA_PATH = Path('~/workspace/data/lipid/GIXD').expanduser()
PLOT_PATH = Path('~/workspace/plot/lipid/GIXD').expanduser()

# Define dataset configurations
DATASET_CONFIGS = [
    {
        'name': 'azotrans',
        'indices': [54, 58, 62],
        'pressures': [10, 20, 30],
    },
    {
        'name': 'azocis',
        'indices': [78, 82, 86, 90],
        'pressures': [5, 10, 20, 30],
    },
    {
        'name': 'azocis02',
        'indices': [106, 110],
        'pressures': [3.3, 30],
    },
    {
        'name': 'azocis03',
        'indices': [115, 119],
        'pressures': [0.1, 30],
    },
    {
        'name': 'dopc',
        'indices': [16, 12, 20, 24],
        'pressures': [0.1, 10, 20, 30],
    },
    {
        'name': 'redazo',
        'indices': [128, 132, 136, 140],
        'pressures': [0.1, 10, 20, 30],
    },
]

# Water reference configuration
WATER_INDEX = 44
WATER_PRESSURE = 0.4


def main():
    # Load water reference
    water_ref = GIXDData.load(DATA_PATH, 'water', WATER_INDEX, WATER_PRESSURE)

    # Load all datasets and subtract water
    datasets = {}
    for config in DATASET_CONFIGS:
        dataset = GIXDDataset.load(
            DATA_PATH,
            config['name'],
            config['indices'],
            config['pressures'],
        )
        dataset.water_reference = water_ref
        dataset.subtract_water()
        datasets[config['name']] = dataset

    # Create project and plot all
    project = GIXDProject(
        data_path=DATA_PATH,
        plot_path=PLOT_PATH,
        datasets=datasets
    )
    project.plot_all(
        roi_q=[0.7, 2.0, 0, 10],
        roi_theta=[1.25, 1.5, 0, 60],
        fit_q_range=(0.7, 2.0),
        ignore_q_range=[(1.49, 1.53), (1.67, 1.71)],
        fit_theta_range=(0, 50),
        ignore_theta_range=[(0, 5)]
    )

if __name__ == '__main__':
    main()
