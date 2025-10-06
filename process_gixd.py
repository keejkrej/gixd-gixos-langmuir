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
from utils.background import subtract_invquad_background
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

# Multiple qz slice ranges for horizontal slice analysis
QZ_SLICE_RANGES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

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

    # Collect all data for organized saving
    all_2d_data = {}  # Will store 2D maps
    all_1d_data = {}  # Will store 1D profiles
    all_horizontal_slices = {}  # Will store horizontal slices
    da_cart_binned = None

    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i)

        # Preserve the original cartesian data (pre-subtraction)
        da_cart_orig = da_cart.sel(qz=slice(QZ_CUTOFF, None))

        # Bin both qz and qxy early for better SNR
        da_cart_binned = da_cart_orig.coarsen(
            qz=QZ_BIN, qxy=QXY_BIN, boundary="trim"
        ).mean()

        # Store original cartesian data
        all_2d_data[f"{i}_{p}_orig_cart"] = da_cart_binned

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

            # Store water-subtracted data
            all_2d_data[f"{i}_{p}_sub_water_cart"] = da_cart_sub_water
        elif not is_water and SUBTRACT_INVQUAD:
            # Inverse quadratic background subtraction
            try:
                # Fit and subtract inverse quadratic background on binned data
                da_cart_sub_invquad, background, fit_mask = subtract_invquad_background(
                    da_cart_binned,
                    num_fitting_points=INVQUAD_NUM_FITTING_POINTS,
                    use_global_fit=INVQUAD_USE_GLOBAL_FIT,
                )

                # Store inverse quadratic data
                all_2d_data[f"{i}_{p}_sub_invquad_cart"] = da_cart_sub_invquad
                all_2d_data[f"{i}_{p}_bg_invquad_cart"] = background
                # Note: fit_mask is not saved as it's not used in current plotting pipeline

                # Extract horizontal slice comparison using binned original and background
                horizontal_slices_ds = extract_horizontal_slice_comparison(
                    da_cart_binned, background, name, i, p
                )

                if horizontal_slices_ds is not None:
                    # Store horizontal slices for consolidated saving
                    all_horizontal_slices[f"{i}_{p}_horizontal_slices"] = (
                        horizontal_slices_ds
                    )

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

            # Extract I(q) profile
            intensity_q_sub_invquad = extract_intensity_q(
                da_polar_sub_invquad,
                q_range=(ROI_IQ[0], ROI_IQ[1]),
                tau_range=(ROI_IQ[2], ROI_IQ[3]),
            )

            # Extract I(tau) profile
            intensity_tau_sub_invquad = extract_intensity_tau(
                da_polar_sub_invquad,
                q_range=(ROI_ITAU[0], ROI_ITAU[1]),
                tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
            )

            # Store 2D polar and 1D profile data
            all_2d_data[f"{i}_{p}_sub_invquad_polar"] = da_polar_sub_invquad
            all_1d_data[f"{i}_{p}_sub_invquad_Iq"] = intensity_q_sub_invquad
            all_1d_data[f"{i}_{p}_sub_invquad_Itau"] = intensity_tau_sub_invquad

        # Process polar and 1D intensity profiles for the water‑subtracted version
        if da_cart_sub_water is not None:
            da_polar_sub_water = gixd_cartesian2polar(
                da_cart_sub_water, dq=Q_BIN, dtau=TAU_BIN
            )

            # Extract I(q) profile
            intensity_q_sub_water = extract_intensity_q(
                da_polar_sub_water,
                q_range=(ROI_IQ[0], ROI_IQ[1]),
                tau_range=(ROI_IQ[2], ROI_IQ[3]),
            )

            # Extract I(tau) profile
            intensity_tau_sub_water = extract_intensity_tau(
                da_polar_sub_water,
                q_range=(ROI_ITAU[0], ROI_ITAU[1]),
                tau_range=(ROI_ITAU[2], ROI_ITAU[3]),
            )

            # Store water-subtracted data
            all_2d_data[f"{i}_{p}_sub_water_polar"] = da_polar_sub_water
            all_1d_data[f"{i}_{p}_sub_water_Iq"] = intensity_q_sub_water
            all_1d_data[f"{i}_{p}_sub_water_Itau"] = intensity_tau_sub_water

    # Save organized data files consolidated by sample
    if not is_water:
        # Save 2D maps in one file
        if all_2d_data:
            ds_2d = xr.Dataset(all_2d_data)
            ds_2d.attrs = {
                "description": f"2D maps for {name}",
                "sample_name": name,
                "data_types": "cartesian maps, polar maps, background, fit masks",
            }
            ds_2d.to_netcdf(out_dir / f"{name}_2d_maps.nc")

        # Save 1D profiles in one file
        if all_1d_data:
            ds_1d = xr.Dataset(all_1d_data)
            ds_1d.attrs = {
                "description": f"1D intensity profiles for {name}",
                "sample_name": name,
                "data_types": "I(q) and I(tau) profiles",
            }
            ds_1d.to_netcdf(out_dir / f"{name}_1d_profiles.nc")

        # Save consolidated horizontal slices for the entire sample
        if all_horizontal_slices:
            # Create a consolidated dataset with all horizontal slices
            consolidated_horizontal_slices = {}
            for key, ds in all_horizontal_slices.items():
                # Add all variables from each horizontal slice dataset
                for var_name, var_data in ds.data_vars.items():
                    consolidated_horizontal_slices[f"{key}_{var_name}"] = var_data

            if consolidated_horizontal_slices:
                ds_horizontal = xr.Dataset(consolidated_horizontal_slices)
                ds_horizontal.attrs = {
                    "description": f"Horizontal slice profiles for {name}",
                    "sample_name": name,
                    "data_types": "horizontal slice profiles for multiple qz ranges",
                }
                ds_horizontal.to_netcdf(out_dir / f"{name}_horizontal_slices.nc")

    if is_water:
        return da_cart_binned
    return None


def extract_horizontal_slice_comparison(
    da_cart_orig: xr.DataArray,
    da_background: xr.DataArray,
    name: str,
    index: int,
    pressure: float,
) -> Optional[xr.Dataset]:
    """
    Extract horizontal slice (constant qz) profiles for original data and background.

    Takes the mean over multiple qz ranges to create 1D profiles for each range.
    Returns a Dataset containing all slices.

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

    Returns:
    --------
    xr.Dataset or None
        Dataset containing original, background, and difference profiles for all qz ranges
    """
    try:
        # Create lists to store profiles for each qz range
        orig_profiles = []
        bg_profiles = []
        diff_profiles = []
        qz_range_labels = []

        for qz_min, qz_max in QZ_SLICE_RANGES:
            # Select qz range
            da_orig_slice = da_cart_orig.sel(qz=slice(qz_min, qz_max))
            da_bg_slice = da_background.sel(qz=slice(qz_min, qz_max))

            # Average over qz dimension to get 1D profiles
            orig_profile = da_orig_slice.mean(dim="qz")
            bg_profile = da_bg_slice.mean(dim="qz")
            diff_profile = orig_profile - bg_profile

            # Add metadata to profiles
            common_attrs = {
                "description": f"Horizontal slice profile (qz mean: {qz_min}-{qz_max})",
                "sample_name": name,
                "index": index,
                "pressure": pressure,
                "qz_range_min": qz_min,
                "qz_range_max": qz_max,
                "model": "I(qxy,qz) = (A*qz + B)*qxy^-2 + C",
            }
            orig_profile.attrs = {**common_attrs, "profile_type": "original"}
            bg_profile.attrs = {**common_attrs, "profile_type": "background"}
            diff_profile.attrs = {**common_attrs, "profile_type": "difference"}

            orig_profiles.append(orig_profile)
            bg_profiles.append(bg_profile)
            diff_profiles.append(diff_profile)
            qz_range_labels.append(f"{qz_min:.1f}_{qz_max:.1f}")

        # Create Dataset with all profiles
        data_vars = {}
        for i, (orig, bg, diff, label) in enumerate(
            zip(orig_profiles, bg_profiles, diff_profiles, qz_range_labels)
        ):
            data_vars[f"original_qz_{label}"] = orig
            data_vars[f"background_qz_{label}"] = bg
            data_vars[f"difference_qz_{label}"] = diff

        # Create Dataset
        ds = xr.Dataset(data_vars)

        # Add global attributes
        ds.attrs = {
            "description": f"Horizontal slice profiles for {name} idx={index} p={pressure}",
            "sample_name": name,
            "index": index,
            "pressure": pressure,
            "qz_ranges": str(QZ_SLICE_RANGES),
            "model": "I(qxy,qz) = (A*qz + B)*qxy^-2 + C",
        }

        return ds

    except Exception as e:
        print(
            f"Warning: Failed to extract horizontal slice profiles for {name}_{index}_{pressure}: {e}"
        )
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
