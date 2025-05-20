from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from xrd_anlyutils.math import transform
from xrd_anlyutils.math import roi

@dataclass
class GIXDData:
    """Class representing a single GIXD measurement."""
    name: str
    pressure: float
    index: int
    intensity: np.ndarray
    qxy: np.ndarray
    qz: np.ndarray
    
    @classmethod
    def from_files(cls, path: Path, name: str, index: int, pressure: float) -> 'GIXDData':
        """
        Create a GIXDData instance from data files.
        
        Args:
            path: Base directory path
            name: Base name for files
            index: Index number
            pressure: Pressure value
            
        Returns:
            GIXDData instance
        """
        data_path = Path(path) / name
        I = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_I.dat')
        Qxy = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_Qxy.dat')
        Qz = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_Qz.dat')
        I = np.nan_to_num(I, nan=0)
        return cls(name=name, pressure=pressure, index=index, intensity=I, qxy=Qxy, qz=Qz)

@dataclass
class GIXDDataset:
    """Class representing a collection of GIXD measurements."""
    name: str
    measurements: List[GIXDData]
    water_reference: Optional[GIXDData] = None
    
    def subtract_water(self) -> None:
        """Subtract water reference from all measurements."""
        if self.water_reference is not None:
            for measurement in self.measurements:
                measurement.intensity -= self.water_reference.intensity
    
    @classmethod
    def from_config(cls, path: Path, name: str, indices: List[int], pressures: List[float], water_index: Optional[int] = None, water_pressure: Optional[float] = None) -> 'GIXDDataset':
        """
        Create a GIXDDataset instance from configuration.
        
        Args:
            path: Base directory path
            name: Base name for files
            indices: List of index numbers
            pressures: List of pressure values
            water_index: Optional water reference index
            water_pressure: Optional water reference pressure
            
        Returns:
            GIXDDataset instance
        """
        measurements = [GIXDData.from_files(path, name, idx, press) 
                       for idx, press in zip(indices, pressures)]
        water_ref = None
        if water_index is not None:
            water_ref = GIXDData.from_files(path, 'water', water_index, water_pressure)
        return cls(name=name, measurements=measurements, water_reference=water_ref)

@dataclass
class GIXDProject:
    """Class representing a complete GIXD project with multiple datasets."""
    data_path: Path
    plot_path: Path
    datasets: Dict[str, GIXDDataset]
    
    def plot_gixd(self, measurement: GIXDData, vmin: Optional[float] = None, 
                 vmax: Optional[float] = None, savepath: Optional[Path] = None) -> None:
        """
        Plot GIXD data in both Cartesian and polar coordinates.
        
        Args:
            measurement: GIXDData instance to plot
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            savepath: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        im1 = ax1.imshow(measurement.intensity, cmap='inferno', 
                        extent=[np.min(measurement.qxy), np.max(measurement.qxy), 
                               np.min(measurement.qz), np.max(measurement.qz)], 
                        origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax1.set_xlabel('Qxy [A$^{-1}$]')
        ax1.set_ylabel('Qz [A$^{-1}$]')
        ax1.set_title('Cartesian')
        plt.colorbar(im1, ax=ax1)

        I_polar, Q, theta = transform.cartesian2polar(measurement.intensity, 
                                                    measurement.qxy, measurement.qz, 0.01, 0.01)

        im2 = ax2.imshow(I_polar, cmap='inferno', 
                        extent=[np.min(Q), np.max(Q), 
                               np.rad2deg(np.min(theta)), np.rad2deg(np.max(theta))], 
                        origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax2.set_xlabel('Q [A$^{-1}$]')
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
    
    def plot_peak_profile(self, dataset: GIXDDataset, savepath: Optional[Path] = None) -> None:
        """
        Plot peak profiles for all measurements in a dataset.
        
        Args:
            dataset: GIXDDataset instance to plot
            savepath: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        for measurement in dataset.measurements:
            I_polar, Q, theta = transform.cartesian2polar(measurement.intensity, 
                                                        measurement.qxy, measurement.qz, 0.01, 0.01)
            ax1.plot(Q, roi.get_avg([0.7, 1.8, 0, 0.2], I_polar, Q, theta, axis='y'), 
                    label=f'{measurement.pressure}')
            ax2.plot(np.rad2deg(theta), 
                    roi.get_avg([1.25, 1.5, 0, 0.8], I_polar, Q, theta, axis='x'), 
                    label=f'{measurement.pressure}')
            
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
    
    def plot_all(self) -> None:
        """Plot all datasets in the project."""
        for dataset in self.datasets.values():
            for measurement in dataset.measurements:
                self.plot_gixd(measurement, vmin=0, vmax=10, 
                             savepath=self.plot_path / dataset.name)
            self.plot_peak_profile(dataset, savepath=self.plot_path / dataset.name) 