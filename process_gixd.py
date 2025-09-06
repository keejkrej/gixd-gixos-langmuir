from pathlib import Path
from utils.data.gixd import (
    load_gixd_xarray,
    gixd_cartesian2polar,
    extract_intensity_q,
    extract_intensity_theta,
    remove_peaks_from_1d,
)
import xarray as xr
from data_gixd import Sample, SAMPLES, WATER

DATA_DIR = "data/gixd"
PROCESSED_DIR = "processed/gixd"
ROI_Q = [0.7, 2.0, 0, 10]  # [q_min, q_max, theta_min, theta_max]
ROI_THETA = [1.25, 1.5, 0, 50]  # [q_min, q_max, theta_min, theta_max]


def process_sample(
    data_dir: Path, processed_dir: Path, data: Sample, da_water: xr.DataArray
):
    name, index, pressure = data["name"], data["index"], data["pressure"]
    (processed_dir / name).mkdir(parents=True, exist_ok=True)

    print(f"Processing {name}...")
    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i) - da_water
        da_cart.to_netcdf(processed_dir / name / f"{name}_{i}_{p}_cartesian.nc")

        da_polar = gixd_cartesian2polar(da_cart, dr=0.005, dtheta=0.005)
        da_polar.to_netcdf(processed_dir / name / f"{name}_{i}_{p}_polar.nc")

        intensity_q = extract_intensity_q(
            da_polar, q_range=(ROI_Q[0], ROI_Q[1]), theta_range=(ROI_Q[2], ROI_Q[3])
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

        intensity_theta = extract_intensity_theta(
            da_polar,
            q_range=(ROI_THETA[0], ROI_THETA[1]),
            theta_range=(ROI_THETA[2], ROI_THETA[3]),
        )
        # Remove peaks from I(theta) profile
        intensity_theta_no_peaks, theta_coords_filtered = remove_peaks_from_1d(
            intensity_theta.values,
            intensity_theta.theta.values,
            window_size=21,
            sigma_threshold=2.0,
            exclusion_radius=2,
        )
        intensity_theta_clean = xr.DataArray(
            intensity_theta_no_peaks,
            dims=["theta"],
            coords={"theta": theta_coords_filtered},
            name="intensity_theta",
        )
        intensity_theta_clean.to_netcdf(
            processed_dir / name / f"{name}_{i}_{p}_intensity_theta.nc"
        )


def main():
    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    water_name, water_index = WATER["name"], WATER["index"][0]
    da_water = load_gixd_xarray(Path(DATA_DIR), water_name, water_index)
    print(f"Loaded water reference: {water_name}_{water_index}.")

    for s in SAMPLES:
        process_sample(Path(DATA_DIR), processed_dir, s, da_water)
    print("GIXD processing completed.")


if __name__ == "__main__":
    main()
