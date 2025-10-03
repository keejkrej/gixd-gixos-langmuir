"""
GIXD Data Processing Pipeline

Background subtraction options:
- SUBTRACT_WATER: Use water reference subtraction (requires water reference data)
- SUBTRACT_INVQUAD: Use inverse quadratic background subtraction: I = A*Qxy^-2 + B
  Only one method can be enabled at a time - they are mutually exclusive.
"""

from pathlib import Path
from typing import Optional, Tuple

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
QZ_BIN = 5  # channels
QXY_BIN = 5  # channels for qxy binning before background fitting
Q_BIN = 0.05
TAU_BIN = 0.02

# Inverse quadratic background subtraction settings with global fitting
# Uses model: I(qxy,qz) = (A*qz + B)*Qxy^-2 + C fitted globally across all slices
# Edges are automatically detected by scanning for jumps from zero-filled regions
INVQUAD_NUM_FITTING_POINTS = 10  # Points to use from each edge for fitting
INVQUAD_USE_GLOBAL_FIT = True

QZ_SLICE_RANGE = (0.2, 0.4)

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

    da_cart_binned = None

    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i)

        # Preserve the original cartesian data (pre-subtraction)
        da_cart_orig = da_cart.sel(qz=slice(QZ_CUTOFF, None))

        # Bin both qz and qxy early for better SNR
        da_cart_binned = da_cart_orig.coarsen(
            qz=QZ_BIN, qxy=QXY_BIN, boundary="trim"
        ).mean()
        da_cart_binned.to_netcdf(out_dir / f"{name}_{i}_{p}_orig_cart.nc")

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
            # Water subtraction on binned data
            da_cart_sub_water = da_cart_binned - da_water  # Both already binned

            da_cart_sub_water.to_netcdf(out_dir / f"{name}_{i}_{p}_sub_water_cart.nc")
        elif not is_water and SUBTRACT_INVQUAD:
            # Inverse quadratic background subtraction
            try:
                # Fit and subtract inverse quadratic background on binned data
                da_cart_sub_invquad, background, fit_mask = subtract_invquad_background(
                    da_cart_binned,
                    num_fitting_points=INVQUAD_NUM_FITTING_POINTS,
                    use_global_fit=INVQUAD_USE_GLOBAL_FIT,
                )

                # Save the subtracted data (already binned)
                da_cart_sub_invquad.to_netcdf(
                    out_dir / f"{name}_{i}_{p}_sub_invquad_cart.nc"
                )

                # Save the background (already binned)
                background.to_netcdf(out_dir / f"{name}_{i}_{p}_bg_invquad_cart.nc")

                # Save the fit mask
                fit_mask.to_netcdf(out_dir / f"{name}_{i}_{p}_fit_mask_cart.nc")

                # Extract horizontal slice comparison using binned original and background
                orig_profile, bg_profile = extract_horizontal_slice_comparison(
                    da_cart_binned, background, name, i, p
                )

                if orig_profile is not None:
                    out_base = f"{name}_{i}_{p}_horizontal_slice"
                    orig_file = out_dir / f"{out_base}_orig.nc"
                    bg_file = out_dir / f"{out_base}_bg.nc"
                    diff_profile = orig_profile - bg_profile
                    diff_file = out_dir / f"{out_base}_diff.nc"

                    orig_profile.to_netcdf(orig_file)
                    bg_profile.to_netcdf(bg_file)
                    diff_profile.to_netcdf(diff_file)

            except Exception as e:
                print(
                    f"Warning: Failed to subtract inverse quadratic background for {name}_{i}_{p}: {e}"
                )
                da_cart_sub_invquad = None

        # Process polar and 1D intensity profiles for the inverse quadratic subtracted version
        if da_cart_sub_invquad is not None:
            da_polar_sub_invquad = gixd_cartesian2polar(
                da_cart_sub_invquad,
                dq=Q_BIN,
                dtau=TAU_BIN,  # Uses binned cartesian
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
            da_polar_sub_water.to_netcdf(out_dir / f"{name}_{i}_{p}_sub_water_polar.nc")

            # Extract I(q) profile
            intensity_q_sub_water = extract_intensity_q(
                da_polar_sub_water,
                q_range=(ROI_IQ[0], ROI_IQ[1]),
                tau_range=(ROI_IQ[2], ROI_IQ[3]),
            )
            intensity_q_sub_water.to_netcdf(out_dir / f"{name}_{i}_{p}_sub_water_Iq.nc")

            # Extract I(tau) profile
            intensity_tau_sub_water = extract_intensity_tau(
                da_polar_sub_water,
                q_range=(ROI_ITAU[0], ROI_ITAU[1]),
                tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
            )
            intensity_tau_sub_water.to_netcdf(
                out_dir / f"{name}_{i}_{p}_sub_water_Itau.nc"
            )

    if is_water:
        return da_cart_binned
    return None


def extract_horizontal_slice_comparison(
    da_cart_orig: xr.DataArray,
    da_background: xr.DataArray,
    name: str,
    index: int,
    pressure: float,
    qz_range: tuple[float, float] = QZ_SLICE_RANGE,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Extract horizontal slice (constant qz) profiles for original data and background.

    Takes the mean over a small qz range near 0 to create 1D profiles.

    Parameters:
    -----------
    da_cart_orig : xr.DataArray
        Original unsubtracted cartesian data
    da_background : xr.DataArray
        Fitted background data
    name : str
        Sample name
    index : int
        Sample index
    pressure : float
        Sample pressure
    qz_range : tuple[float, float]
        qz range to average over (default: 0.2 to 0.4)

    Returns:
    --------
    tuple[xr.DataArray, xr.DataArray] or (None, None)
        Original profile and background profile
    """
    try:
        # Select qz range near 0
        da_orig_slice = da_cart_orig.sel(qz=slice(qz_range[0], qz_range[1]))
        da_bg_slice = da_background.sel(qz=slice(qz_range[0], qz_range[1]))

        # Average over qz dimension to get 1D profiles
        orig_profile = da_orig_slice.mean(dim="qz")
        bg_profile = da_bg_slice.mean(dim="qz")

        # Add metadata to profiles
        common_attrs = {
            "description": f"Horizontal slice profile (qz mean: {qz_range[0]}-{qz_range[1]})",
            "sample_name": name,
            "index": index,
            "pressure": pressure,
            "qz_range_min": qz_range[0],
            "qz_range_max": qz_range[1],
            "model": "I(qxy,qz) = (A*qz + B)*qxy^-2 + C",
        }
        orig_profile.attrs = {**common_attrs, "profile_type": "original"}
        bg_profile.attrs = {**common_attrs, "profile_type": "background"}

        return orig_profile, bg_profile

    except Exception as e:
        print(
            f"Warning: Failed to extract horizontal slice profiles for {name}_{index}_{pressure}: {e}"
        )
        return None, None


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
