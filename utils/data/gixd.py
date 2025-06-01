from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from utils.fit.gixd import fit_lorentzian, fit_gaussian
from utils.math import transform, roi

@dataclass
class GIXDData:
    """Class representing a single Langmuir GIXD measurement."""
    intensity: np.ndarray
    qxy: np.ndarray
    qz: np.ndarray
    name: str
    index: int
    pressure: float

    @classmethod
    def load(
        cls,
        path: Path,
        name: str,
        index: int,
        pressure: float
    ) -> 'GIXDData':
        data_path = Path(path) / name
        intensity = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_I.dat')
        qxy = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_Qxy.dat')
        qz = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_Qz.dat')
        intensity = np.nan_to_num(intensity, nan=0)
        return cls(
            intensity=intensity,
            qxy=qxy,
            qz=qz,
            name=name,
            index=index,
            pressure=pressure
        )

@dataclass
class GIXDDataset:
    """Class representing a collection of Langmuir GIXD measurements."""
    name: str
    measurements: List[GIXDData]
    water_reference: GIXDData | None = None

    def subtract_water(self) -> None:
        """Subtract water reference from all measurements."""
        if self.water_reference is not None:
            for measurement in self.measurements:
                measurement.intensity -= self.water_reference.intensity
        else:
            print("No water reference provided.")

    @classmethod
    def load(
        cls,
        path: Path,
        name: str,
        indices: List[int],
        pressures: List[float]
    ) -> 'GIXDDataset':
        measurements = [GIXDData.load(path, name, idx, press)
                       for idx, press in zip(indices, pressures)]
        return cls(
            name=name,
            measurements=measurements
        )

@dataclass
class GIXDProject:
    """Class representing a complete GIXD project with multiple datasets."""
    data_path: Path
    plot_path: Path
    datasets: Dict[str, GIXDDataset]

    def plot_2d(
        self,
        measurement: GIXDData,
        vmin: float,
        vmax: float,
        savepath: Path
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        im1 = ax1.imshow(
            measurement.intensity,
            cmap='inferno',
            extent=[np.min(measurement.qxy), np.max(measurement.qxy),
                    np.min(measurement.qz), np.max(measurement.qz)],
            origin='lower',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        ax1.set_xlabel('Qxy [A$^{-1}$]')
        ax1.set_ylabel('Qz [A$^{-1}$]')
        ax1.set_title('Cartesian')
        plt.colorbar(im1, ax=ax1)

        intensity_polar, q, theta = transform.cartesian2polar(
            measurement.intensity,
            measurement.qxy,
            measurement.qz,
            0.01,
            0.01
        )
        theta_deg = np.rad2deg(theta)
        im2 = ax2.imshow(
            intensity_polar,
            cmap='inferno',
            extent=[np.min(q), np.max(q),
                    np.min(theta_deg), np.max(theta_deg)],
            origin='lower',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        ax2.set_xlabel('q [A$^{-1}$]')
        ax2.set_ylabel('theta [deg]')
        ax2.set_title('Polar Rebinning')
        plt.colorbar(im2, ax=ax2)

        fig.suptitle(f'{measurement.name} {measurement.pressure}')
        if savepath is not None:
            if not savepath.exists():
                savepath.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepath / f'{measurement.name}_{measurement.pressure}.png')
            plt.close(fig)
        else:
            plt.show()

    def plot_peak(
        self,
        dataset: GIXDDataset,
        savepath: Path,
        roi_q,
        roi_theta,
        fit_q_range: Tuple[float, float],
        ignore_q_range: List[Tuple[float, float]],
        fit_theta_range: Tuple[float, float],
        ignore_theta_range: List[Tuple[float, float]]
    ) -> None:
        color_palette = plt.get_cmap('tab10')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        offset = 5
        for i, measurement in enumerate(dataset.measurements):
            intensity_polar, q, theta = transform.cartesian2polar(
                measurement.intensity,
                measurement.qxy,
                measurement.qz,
                0.01,
                0.01,
            )
            theta_deg = np.rad2deg(theta)
            intensity_q = roi.get_avg(
                roi_q,
                intensity_polar,
                q,
                theta_deg,
                axis='y'
            )
            ax1.plot(
                q,
                intensity_q + offset * i,
                label=f'{measurement.pressure}',
                color=color_palette(i)
            )

            # Fit and plot Q profile
            q_masked, intensity_q_fit, q_max_fit = fit_lorentzian(q,
                intensity_q,
                fit_range=fit_q_range,
                ignore_range=ignore_q_range
            )
            ax1.plot(
                q_masked,
                intensity_q_fit + offset * i,
                '--',
                color=color_palette(i)
            )

            ax1.axvline(
                q_max_fit,
                color=color_palette(i),
                linestyle='--',
                label=f'{q_max_fit:.2f} A$^{-1}$'
            )

            # theta profile
            intensity_theta = roi.get_avg(
                roi_theta,
                intensity_polar,
                q,
                theta_deg,
                axis='x'
            )
            ax2.plot(
                theta_deg,
                intensity_theta + offset * i,
                label=f'{measurement.pressure}', color=color_palette(i)
            )

            # Fit and plot theta profile
            theta_masked, intensity_theta_fit, theta_max_fit = fit_gaussian(
                theta_deg,
                intensity_theta,
                fit_range=fit_theta_range,
                ignore_range=ignore_theta_range
            )
            ax2.plot(
                theta_masked,
                intensity_theta_fit + offset * i,
                '--',
                color=color_palette(i)
            )
            ax2.axvline(
                theta_max_fit,
                color=color_palette(i),
                linestyle='--',
                label=f'{theta_max_fit:.2f} deg'
            )
        ax1.legend()
        ax1.set_xlabel('Q [A$^{-1}$]')
        ax1.set_ylabel('Intensity [a.u.]')
        ax1.set_title(dataset.name)
        ax2.legend()
        ax2.set_xlabel('theta [deg]')
        ax2.set_ylabel('Intensity [a.u.]')
        ax2.set_title(dataset.name)
        if savepath is not None:
            if not savepath.exists():
                savepath.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepath / f'{dataset.name}_profile.png')
            plt.close(fig)
        else:
            plt.show()

    def plot_all(
        self,
        roi_q: List[float],
        roi_theta: List[float],
        fit_q_range: Tuple[float, float],
        ignore_q_range: List[Tuple[float, float]],
        fit_theta_range: Tuple[float, float],
        ignore_theta_range: List[Tuple[float, float]]
    ) -> None:
        """Plot all datasets in the project."""
        for dataset in self.datasets.values():
            for measurement in dataset.measurements:
                self.plot_2d(
                    measurement,
                    vmin=0,
                    vmax=10,
                    savepath=self.plot_path / dataset.name
                )
            self.plot_peak(
                dataset,
                savepath=self.plot_path / dataset.name,
                roi_q=roi_q,
                roi_theta=roi_theta,
                fit_q_range=fit_q_range,
                ignore_q_range=ignore_q_range,
                fit_theta_range=fit_theta_range,
                ignore_theta_range=ignore_theta_range
            )
