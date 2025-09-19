from pathlib import Path
from utils.data.gixd import (
    load_gixd_xarray,
    gixd_cartesian2polar,
    extract_intensity_q,
    extract_intensity_tau,
    remove_peaks_from_1d,
)
import xarray as xr
from data_gixd import Sample, SAMPLES, WATER, ROI_IQ, ROI_ITAU

DATA_DIR = "data/gixd"
PROCESSED_DIR = "processed/gixd"


def process_sample(
    data_dir: Path, processed_dir: Path, data: Sample, da_water: xr.DataArray
):
    name, index, pressure = data["name"], data["index"], data["pressure"]
    (processed_dir / name).mkdir(parents=True, exist_ok=True)

    print(f"Processing {name}...")
    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i) - da_water
        da_cart.to_netcdf(processed_dir / name / f"{name}_{i}_{p}_cartesian.nc")

        da_polar = gixd_cartesian2polar(da_cart, dr=0.005, dtau=0.005)
        da_polar.to_netcdf(processed_dir / name / f"{name}_{i}_{p}_polar.nc")

        intensity_q = extract_intensity_q(
            da_polar, q_range=(ROI_IQ[0], ROI_IQ[1]), tau_range=(ROI_IQ[2], ROI_IQ[3])
        )
        # Remove peaks from I(q) profile
        intensity_q_no_peaks, q_coords_filtered = remove_peaks_from_1d(
            intensity_q.values,
            intensity_q.q.values,
            window_size=21,
            sigma_threshold=2.0,
            exclusion_radius=0.025,
        )
        intensity_q_clean = xr.DataArray(
            intensity_q_no_peaks,
            dims=["q"],
            coords={"q": q_coords_filtered},
            name="intensity_q",
        )
        intensity_q_clean.to_netcdf(
            processed_dir / name / f"{name}_{i}_{p}_intensity_q.nc"
        )

        intensity_tau = extract_intensity_tau(
            da_polar,
            q_range=(ROI_ITAU[0], ROI_ITAU[1]),
            tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
        )
        # Remove peaks from I(tau) profile
        intensity_tau_no_peaks, tau_coords_filtered = remove_peaks_from_1d(
            intensity_tau.values,
            intensity_tau.tau.values,
            window_size=21,
            sigma_threshold=2.0,
            exclusion_radius=2,
        )
        intensity_tau_clean = xr.DataArray(
            intensity_tau_no_peaks,
            dims=["tau"],
            coords={"tau": tau_coords_filtered},
            name="intensity_tau",
        )
        intensity_tau_clean.to_netcdf(
            processed_dir / name / f"{name}_{i}_{p}_intensity_tau.nc"
        )


def process_water_reference(data_dir: Path, processed_dir: Path, water_data: Sample):
    """Process and save water reference data to NetCDF format."""
    name, index = water_data["name"], water_data["index"][0]
    water_dir = processed_dir / name
    water_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing water reference: {name}_{index}...")

    # Load water data
    da_cart = load_gixd_xarray(data_dir, name, index)
    da_cart.to_netcdf(water_dir / f"{name}_{index}_cartesian.nc")

    # Convert to polar coordinates
    da_polar = gixd_cartesian2polar(da_cart, dr=0.005, dtau=0.005)
    da_polar.to_netcdf(water_dir / f"{name}_{index}_polar.nc")

    # Extract intensity profiles
    intensity_q = extract_intensity_q(
        da_polar, q_range=(ROI_IQ[0], ROI_IQ[1]), tau_range=(ROI_IQ[2], ROI_IQ[3])
    )
    # Remove peaks from I(q) profile
    intensity_q_no_peaks, q_coords_filtered = remove_peaks_from_1d(
        intensity_q.values,
        intensity_q.q.values,
        window_size=21,
        sigma_threshold=2.0,
        exclusion_radius=0.025,
    )
    intensity_q_clean = xr.DataArray(
        intensity_q_no_peaks,
        dims=["q"],
        coords={"q": q_coords_filtered},
        name="intensity_q",
    )
    intensity_q_clean.to_netcdf(water_dir / f"{name}_{index}_intensity_q.nc")

    intensity_tau = extract_intensity_tau(
        da_polar,
        q_range=(ROI_ITAU[0], ROI_ITAU[1]),
        tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
    )
    # Remove peaks from I(tau) profile
    intensity_tau_no_peaks, tau_coords_filtered = remove_peaks_from_1d(
        intensity_tau.values,
        intensity_tau.tau.values,
        window_size=21,
        sigma_threshold=2.0,
        exclusion_radius=2,
    )
    intensity_tau_clean = xr.DataArray(
        intensity_tau_no_peaks,
        dims=["tau"],
        coords={"tau": tau_coords_filtered},
        name="intensity_tau",
    )
    intensity_tau_clean.to_netcdf(water_dir / f"{name}_{index}_intensity_tau.nc")

    return da_cart


def main():
    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process water reference first
    da_water = process_water_reference(Path(DATA_DIR), processed_dir, WATER)

    for s in SAMPLES:
        process_sample(Path(DATA_DIR), processed_dir, s, da_water)
    print("GIXD processing completed.")


if __name__ == "__main__":
    main()
