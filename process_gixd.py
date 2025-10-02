"""
GIXD Data Processing Pipeline

Background subtraction options:
- SUBTRACT_WATER: Use water reference subtraction (requires water reference data)
- SUBTRACT_INVQUAD: Use inverse quadratic background subtraction: I = A*Qxy^-2 + B
  Only one method can be enabled at a time - they are mutually exclusive.
"""

from pathlib import Path
from typing import Optional

from utils.data.gixd import (
    load_gixd_xarray,
    gixd_cartesian2polar,
    extract_intensity_q,
    extract_intensity_tau,
)

# Inverse quadratic background subtraction
# Uses model I = A*Qxy^-2 + B fitted to edge points
# This is an alternative to water reference subtraction
from utils.background import (
    subtract_invquad_background,
)
import xarray as xr
from data_gixd import Sample, get_samples, WATER, ROI_IQ, ROI_ITAU

DATA_DIR = "data/gixd"
PROCESSED_DIR = "processed/gixd"
SUBTRACT_WATER = False
SUBTRACT_INVQUAD = True
IS_TEST = False
QZ_CUTOFF = 0.04
QZ_BIN = 20  # channels
Q_BIN = 0.05
TAU_BIN = 0.0872665

# Inverse quadratic background subtraction settings with slice-wise fitting
# Uses model: I = A(qz)*Qxy^-2 + B(qz) fitted slice-by-slice
# Edges are automatically detected by scanning for jumps from zero-filled regions
INVQUAD_NUM_FITTING_POINTS = 5  # Points to use from each edge for fitting

# Validate subtraction settings (prevent both being enabled)
if SUBTRACT_WATER and SUBTRACT_INVQUAD:
    raise ValueError(
        "Cannot enable both SUBTRACT_WATER and SUBTRACT_INVQUAD. "
        "Choose only one subtraction method: set either SUBTRACT_WATER=True OR SUBTRACT_INVQUAD=True, not both."
    )


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

    print(f"Processing {name}...")

    last_cartesian: Optional[xr.DataArray] = None
    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i)

        # Preserve the original cartesian data (pre-subtraction)
        da_cart_orig = da_cart.sel(qz=slice(QZ_CUTOFF, None))

        # Bin and save the original cartesian data
        da_cart_orig_bin = da_cart_orig.coarsen(qz=QZ_BIN, boundary="trim").mean()
        da_cart_orig_bin.to_netcdf(out_dir / f"{name}_{i}_{p}_orig_cart.nc")

        # Apply background subtraction based on configuration
        da_cart_sub_water = None
        da_cart_sub_invquad = None

        # Either subtract water reference OR subtract inverse quadratic background, not both
        if (
            not is_water
            and SUBTRACT_WATER
            and da_water is not None
            and not SUBTRACT_INVQUAD
        ):
            # Water subtraction
            da_cart_sub_water = da_cart_orig - da_water
            da_cart_sub_water_bin = da_cart_sub_water.coarsen(
                qz=QZ_BIN, boundary="trim"
            ).mean()

            da_cart_sub_water_bin.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_water_cart.nc"
            )
        elif not is_water and SUBTRACT_INVQUAD:
            # Inverse quadratic background subtraction
            try:
                # Fit and subtract inverse quadratic background slice-by-slice
                da_cart_sub_invquad, background = subtract_invquad_background(
                    da_cart_orig,
                    num_fitting_points=INVQUAD_NUM_FITTING_POINTS,
                    return_background=True,
                )

                # Bin and save the inverse quadratic subtracted data
                da_cart_sub_invquad_bin = da_cart_sub_invquad.coarsen(
                    qz=QZ_BIN, boundary="trim"
                ).mean()
                da_cart_sub_invquad_bin.to_netcdf(
                    out_dir / f"{name}_{i}_{p}_sub_invquad_cart.nc"
                )

                # Save the binned background for inspection
                background_bin = background.coarsen(qz=QZ_BIN, boundary="trim").mean()
                background_bin.to_netcdf(
                    out_dir / f"{name}_{i}_{p}_bg_invquad_cart.nc"
                )

            except Exception as e:
                print(
                    f"Warning: Failed to subtract inverse quadratic background for {name}_{i}_{p}: {e}"
                )
                da_cart_sub_invquad = None

        # Process polar and 1D intensity profiles for the inverse quadratic subtracted version
        if da_cart_sub_invquad is not None:
            da_polar_sub_invquad = gixd_cartesian2polar(
                da_cart_sub_invquad, dq=Q_BIN, dtau=TAU_BIN
            )
            # Save full 2D polar data for plotting
            da_polar_sub_invquad.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_invquad_polar.nc"
            )

            # Extract I(q) profile
            intensity_q_sub_invquad = extract_intensity_q(
                da_polar_sub_invquad,
                q_range=(ROI_IQ[0], ROI_IQ[1]),
                tau_range=(ROI_IQ[2], ROI_IQ[3]),
            )
            intensity_q_sub_invquad.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_invquad_Iq.nc"
            )

            # Extract I(tau) profile
            intensity_tau_sub_invquad = extract_intensity_tau(
                da_polar_sub_invquad,
                q_range=(ROI_ITAU[0], ROI_ITAU[1]),
                tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
            )
            intensity_tau_sub_invquad.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_invquad_Itau.nc"
            )

        # Process polar and 1D intensity profiles for the water‑subtracted version
        if da_cart_sub_water is not None:
            da_polar_sub_water = gixd_cartesian2polar(
                da_cart_sub_water, dq=Q_BIN, dtau=TAU_BIN
            )
            # Save full 2D polar data for plotting
            da_polar_sub_water.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_water_polar.nc"
            )

            # Extract I(q) profile
            intensity_q_sub_water = extract_intensity_q(
                da_polar_sub_water,
                q_range=(ROI_IQ[0], ROI_IQ[1]),
                tau_range=(ROI_IQ[2], ROI_IQ[3]),
            )
            intensity_q_sub_water.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_water_Iq.nc"
            )

            # Extract I(tau) profile
            intensity_tau_sub_water = extract_intensity_tau(
                da_polar_sub_water,
                q_range=(ROI_ITAU[0], ROI_ITAU[1]),
                tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
            )
            intensity_tau_sub_water.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_water_Itau.nc"
            )

        # Store the last processed cartesian (original version) for water reference return
        last_cartesian = da_cart_orig

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
