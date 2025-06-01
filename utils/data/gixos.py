from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from utils.fit.gixos import fit_gaussian

@dataclass
class GIXOSData:
    sf: np.ndarray
    qz: np.ndarray
    name: str
    index: int
    pressure: float
    peak_1: Tuple[float, float]
    peak_2: Tuple[float, float]

    @classmethod
    def load(
        cls,
        path: Path,
        name: str,
        index: int,
        pressure: float,
        peak_1: Tuple[float, float],
        peak_2: Tuple[float, float]
    ) -> 'GIXOSData':
        data_path = Path(path) / name
        data = np.loadtxt(data_path / f'{name}_{index:05d}_SF.dat', skiprows=30)
        qz = data[:, 0]
        sf = data[:, 1]
        return cls(
            sf=sf,
            qz=qz,
            name=name,
            index=index,
            pressure=pressure,
            peak_1=peak_1,
            peak_2=peak_2
        )
    
@dataclass
class GIXOSDataset:
    name: str
    measurements: List[GIXOSData]
    peak_1_list: List[float] | None = None
    peak_2_list: List[float] | None = None

    @classmethod
    def load(cls,
            path: Path,
            name: str,
            indices: List[int],
            pressures: List[float],
            peak_1: List[Tuple[float, float]],
            peak_2: List[Tuple[float, float]]
        ) -> 'GIXOSDataset':
        measurements = [GIXOSData.load(path, name, idx, press, peak_1, peak_2) for idx, press, peak_1, peak_2 in zip(indices, pressures, peak_1, peak_2)]
        return cls(
            name=name,
            measurements=measurements
        )
    
@dataclass
class GIXOSProject:
    data_path: Path
    plot_path: Path
    datasets: Dict[str, GIXOSDataset]

    def plot_one(
        self,
        dataset: GIXOSDataset,
        savepath: Path | None = None,
    ) -> None:
        color_palette = plt.get_cmap('tab10')
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        dataset.peak_1_list = []
        dataset.peak_2_list = []
        for i, measurement in enumerate(dataset.measurements):
            ax.scatter(measurement.qz, measurement.sf, label=f'{measurement.pressure}', s=50, alpha=0.5, color=color_palette(i))
            x_masked, intensity_fit, x_max_fit = fit_gaussian(measurement.qz, measurement.sf, measurement.peak_1)
            x_masked_2, intensity_fit_2, x_max_fit_2 = fit_gaussian(measurement.qz, measurement.sf, measurement.peak_2)
            ax.plot(x_masked, intensity_fit, color=color_palette(i))
            ax.plot(x_masked_2, intensity_fit_2, color=color_palette(i))
            ax.axvline(x_max_fit, color=color_palette(i), linestyle='--', label=f'peak 1 = {x_max_fit:.2f}')
            ax.axvline(x_max_fit_2, color=color_palette(i), linestyle='--', label=f'peak 2 = {x_max_fit_2:.2f}')
            dataset.peak_1_list.append(x_max_fit)
            dataset.peak_2_list.append(x_max_fit_2)

        ax.legend()
        ax.set_xlabel('Qz [A$^{-1}$]')
        ax.set_ylabel('SF [a.u.]')
        ax.set_title(dataset.name)
        if savepath is not None:
            if not savepath.exists():
                savepath.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepath / f'{dataset.name}.png')
            plt.close(fig)
        else:
            plt.show()

    def plot_all(
        self,
    ) -> None:
        color_palette = plt.get_cmap('tab10')
        for dataset in self.datasets.values():
            self.plot_one(dataset, self.plot_path / dataset.name)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for i, dataset in enumerate(self.datasets.values()):
            pressure_list = []
            for measurement in dataset.measurements:
                pressure_list.append(measurement.pressure)
            if dataset.peak_1_list is not None and dataset.peak_2_list is not None:
                ax1.plot(pressure_list, dataset.peak_1_list, color=color_palette(i), label=dataset.name, marker='o')
                ax2.plot(pressure_list, dataset.peak_2_list, color=color_palette(i), label=dataset.name, marker='o')
            else:
                continue
        ax1.set_xlabel('pressure [mN/m]')
        ax1.set_ylabel('peak [A$^{-1}$]')
        ax1.set_title('peak 1')
        ax1.legend()
        ax2.set_xlabel('pressure [mN/m]')
        ax2.set_ylabel('peak [A$^{-1}$]')
        ax2.set_title('peak 2')
        ax2.legend()
        plt.savefig(self.plot_path / f'peak.png')
        plt.close(fig)
        