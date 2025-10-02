from pathlib import Path
from typing import Optional

from utils.data.gixd import (
    load_gixd_xarray,
    gixd_cartesian2polar,
    extract_intensity_q,
    extract_intensity_tau,
)
import xarray as xr
from data_gixd import Sample, get_samples, WATER, ROI_IQ, ROI_ITAU

DATA_DIR = "data/gixd"
PROCESSED_DIR = "processed/gixd"
SUBTRACT_WATER = False
IS_TEST = False
QZ_CUTOFF = 0.04
QZ_BIN = 20  # channels
Q_BIN = 0.05
TAU_BIN = 0.0872665


def process_sample(
    data_dir: Path,
    processed_dir: Path,
    data: Sample,
    da_water: Optional[xr.DataArray] = None,
    is_water: bool = False,
) -> Optional[xr.DataArray]:
    """Process either a regular sample or the water reference.

    The original implementation had two nearly identical functions – one for
    regular samples (``process_sample``) and one for the water reference
    (``process_water_reference``). Both performed the same series of steps:

    * Load the raw cartesian data.
    * Optionally subtract the water reference.
    * Slice, bin and save the cartesian data.
    * Convert to polar coordinates and save.
    * Extract ``I(q)`` and ``I(tau)`` intensity profiles and save them.

    The only differences were:

    * The water reference does not perform water subtraction.
    * It returns the loaded cartesian ``DataArray`` so that downstream samples
      can subtract it if ``SUBTRACT_WATER`` is ``True``.

    This merged function captures the shared logic in a single loop. When
    ``is_water`` is ``True`` the subtraction step is skipped and the final
    ``da_cart`` is returned to the caller. For normal samples the function
    returns ``None``.
    """
    name, index, pressure = data["name"], data["index"], data["pressure"]
    out_dir = processed_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {'water reference: ' if is_water else ''}{name}...")

    last_cartesian: Optional[xr.DataArray] = None
    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i)

        # Preserve the original cartesian data (pre‑water subtraction)
        da_orig = da_cart.sel(qz=slice(QZ_CUTOFF, None))

        # Bin and save the original cartesian data
        da_orig_bin = da_orig.coarsen(qz=QZ_BIN, boundary="trim").mean()
        da_orig_bin.to_netcdf(out_dir / f"{name}_{i}_{p}_orig_cartesian.nc")

        # If this is not the water reference and water subtraction is enabled,
        # create a water‑subtracted version.
        if not is_water and SUBTRACT_WATER and da_water is not None:
            da_sub = da_orig - da_water
            da_sub_bin = da_sub.coarsen(qz=QZ_BIN, boundary="trim").mean()
            da_sub_bin.to_netcdf(out_dir / f"{name}_{i}_{p}_sub_cartesian.nc")
        else:
            da_sub = None

        # -----------------------------------------------------------------
        # Process polar and intensity data for the original cartesian data
        da_polar_orig = gixd_cartesian2polar(da_orig, dq=Q_BIN, dtau=TAU_BIN)
        da_polar_orig.to_netcdf(out_dir / f"{name}_{i}_{p}_orig_polar.nc")
        intensity_q_orig = extract_intensity_q(
            da_polar_orig,
            q_range=(ROI_IQ[0], ROI_IQ[1]),
            tau_range=(ROI_IQ[2], ROI_IQ[3]),
        )
        intensity_q_orig.to_netcdf(out_dir / f"{name}_{i}_{p}_orig_intensity_q.nc")
        intensity_tau_orig = extract_intensity_tau(
            da_polar_orig,
            q_range=(ROI_ITAU[0], ROI_ITAU[1]),
            tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
        )
        intensity_tau_orig.to_netcdf(out_dir / f"{name}_{i}_{p}_orig_intensity_tau.nc")

        # Process polar and intensity data for the water‑subtracted version, if it exists
        if da_sub is not None:
            da_polar_sub = gixd_cartesian2polar(da_sub, dq=Q_BIN, dtau=TAU_BIN)
            da_polar_sub.to_netcdf(out_dir / f"{name}_{i}_{p}_sub_polar.nc")
            intensity_q_sub = extract_intensity_q(
                da_polar_sub,
                q_range=(ROI_IQ[0], ROI_IQ[1]),
                tau_range=(ROI_IQ[2], ROI_IQ[3]),
            )
            intensity_q_sub.to_netcdf(out_dir / f"{name}_{i}_{p}_sub_intensity_q.nc")
            intensity_tau_sub = extract_intensity_tau(
                da_polar_sub,
                q_range=(ROI_ITAU[0], ROI_ITAU[1]),
                tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
            )
            intensity_tau_sub.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_intensity_tau.nc"
            )

        # Store the last processed cartesian (original version) for water reference return
        last_cartesian = da_orig

    if is_water:
        return last_cartesian
    return None


def main():
    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process water reference and capture its cartesian DataArray for later subtraction.
    da_water = process_sample(Path(DATA_DIR), processed_dir, WATER, is_water=True)

    for s in get_samples(IS_TEST):
        process_sample(Path(DATA_DIR), processed_dir, s, da_water)
    print("GIXD processing completed.")


if __name__ == "__main__":
    main()
