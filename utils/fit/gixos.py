import numpy as np
from typing import Tuple, Dict, Any

from refnx.dataset import ReflectDataset
from refnx.reflect import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

# Shared model defaults for azo-PC monolayer (X-ray)
# Units for SLD are x10^-6 Å^-2 in refnx
WATER_SLD = 9.43
TAILS_SLD_INIT = 8.6
HEAD_SLD_INIT = 14.5

# Bounds reflecting azo-PC expectations while allowing refinement
TAILS_THICK_BOUNDS = (10.0, 30.0)
# Tightened SLD bounds around typical values
TAILS_SLD_BOUNDS = (8.0, 9.0)
HEAD_THICK_BOUNDS = (5.0, 20.0)
HEAD_SLD_BOUNDS = (13.5, 15.5)
ROUGH_BOUNDS = (0.5, 6.0)


def create_model() -> Tuple[ReflectModel, Any]:
    """Create a refnx ReflectModel for an azo-PC monolayer.

    Structure: air | tails | head | water
    - Uses shared SLD defaults and loose bounds for thickness and SLDs.
    """
    air = SLD(0.0, name="air")
    water = SLD(WATER_SLD, name="water")
    tails = SLD(TAILS_SLD_INIT, name="tails")
    heads = SLD(HEAD_SLD_INIT, name="heads")

    # initial thickness/roughness; roughness will be varied with ROUGH_BOUNDS
    structure = air | tails(18.0, 2.5) | heads(8.0, 2.5, vfsolv=0.2) | water

    # Roughness on all interfaces
    for i in range(len(structure)):
        structure[i].rough.setp(vary=True, bounds=ROUGH_BOUNDS)

    model = ReflectModel(structure, bkg=0.0, scale=1.0)

    # Parameter bounds
    tails_layer = structure[1]
    heads_layer = structure[2]

    tails_layer.thick.setp(vary=True, bounds=TAILS_THICK_BOUNDS)
    tails_layer.sld.real.setp(vary=True, bounds=TAILS_SLD_BOUNDS)

    heads_layer.thick.setp(vary=True, bounds=HEAD_THICK_BOUNDS)
    heads_layer.sld.real.setp(vary=True, bounds=HEAD_SLD_BOUNDS)
    heads_layer.vfsolv.setp(vary=True, bounds=(0.0, 0.8))

    return model, structure


# ---------- Shared helpers ----------

def _fit_objective(obj: Objective, verbose: bool, maxiter: int) -> float:
    fitter = CurveFitter(obj)
    if verbose:
        print("Stage 1: Differential Evolution")
    fitter.fit(
        "differential_evolution",
        seed=42,
        maxiter=maxiter,
        workers=-1,
        updating="deferred",
    )
    # Nudge parameters away from hard bounds before local refinement
    for p in obj.varying_parameters():
        if hasattr(p.bounds, "lb") and hasattr(p.bounds, "ub"):
            lo, hi = p.bounds.lb, p.bounds.ub
            if p.value <= lo:
                p.value = lo + 0.001 * (hi - lo)
            elif p.value >= hi:
                p.value = hi - 0.001 * (hi - lo)
    if verbose:
        print("Stage 2: Local refinement")
    fitter.fit("least_squares")
    return obj.chisqr()




def _estimate_scale(model: ReflectModel, q: np.ndarray, y: np.ndarray) -> float:
    y_model = model(q)
    q_mid_mask = (q >= 0.08) & (q <= 0.3)
    if q_mid_mask.sum() > 3:
        scale_est = np.median(y[q_mid_mask] / np.maximum(y_model[q_mid_mask], 1e-30))
        return float(np.clip(scale_est, 0.01, 1000))
    return 1.0


# ---------- R (direct reflectivity) pipeline ----------

def _clean_data_r(q, R, dR, dqz):
    mask_physical = (R > 0) & np.isfinite(R) & np.isfinite(dR) & (dR > 0)
    mask_qrange = (q >= 0.025) & (q <= 0.6)
    mask_clean = mask_physical & mask_qrange
    # Remove extreme outliers
    for i in range(1, len(R) - 1):
        if R[i] > 100 * max(R[i - 1], R[i + 1]) and R[i] > 1000:
            mask_clean[i] = False
    return q[mask_clean], R[mask_clean], dR[mask_clean], 2.0 * dqz[mask_clean]


def fit_r(
    r_file: str,
    save_plot: str | None = None,
    show_plot: bool = True,
    verbose: bool = True,
    de_maxiter: int = 200,
) -> Dict[str, Any]:
    if verbose:
        print("=" * 60)
        print("R DATA FITTING")
        print("=" * 60)
    q_r, R_r, dR_r, dqz_r = np.loadtxt(r_file, skiprows=28, unpack=True)
    q, R, dR, dq = _clean_data_r(q_r, R_r, dR_r, dqz_r)

    if verbose:
        print(f"File: {r_file}")
        print(f"Using {len(q)} of {len(q_r)} data points")
        print(f"q range: {q.min():.3f} to {q.max():.3f} Å⁻¹")
        print(f"R range: {R.min():.2e} to {R.max():.2e}")

    ds = ReflectDataset(data=(q, R, dR))
    ds.x_err = dq

    model, structure = create_model()

    scale_est = _estimate_scale(model, q, R)
    model.scale.value = scale_est
    model.scale.setp(vary=True, bounds=(scale_est * 0.1, scale_est * 10))
    model.bkg.setp(vary=True, bounds=(0.0, np.min(R) * 0.5))

    obj = Objective(model, ds)
    chi2 = _fit_objective(obj, verbose=verbose, maxiter=de_maxiter)

    npts = len(q)
    nvary = len(obj.varying_parameters())
    tails_layer = structure[1]
    heads_layer = structure[2]

    results = {
        "method": "R",
        "file": r_file,
        "total_thickness": tails_layer.thick.value + heads_layer.thick.value,
        "tails_thick": tails_layer.thick.value,
        "tails_sld": tails_layer.sld.real.value,
        "heads_thick": heads_layer.thick.value,
        "heads_sld": heads_layer.sld.real.value,
        "heads_vfsolv": heads_layer.vfsolv.value,
        "scale": model.scale.value,
        "bkg": getattr(model, "bkg", 0.0),
        "chi2_red": chi2 / max(1, (npts - nvary)),
        "npts": npts,
        "q_data": q,
        "R_data": R,
        "dR_data": dR,
        "R_fit": model(q),
    }

    # Verbose and plotting are suppressed/unused in process_gixos pipeline.
    return results


# ---------- RFXSF (Chen method, intrinsic structure) pipeline ----------

def _calculate_fresnel_rf(q, rho_water=WATER_SLD * 1e-6):
    k = q / 2.0
    kz_air = k
    kz_water = np.sqrt(k**2 - 4 * np.pi * rho_water + 0j)
    r_fresnel = (kz_air - kz_water) / (kz_air + kz_water)
    RF = np.abs(r_fresnel) ** 2
    return RF


def _create_intrinsic_data(q_sf, SF, dSF, dq_sf):
    RF = _calculate_fresnel_rf(q_sf)
    R_intrinsic = RF * SF
    dR_intrinsic = RF * dSF
    mask = (q_sf >= 0.02) & (q_sf <= 0.8) & (R_intrinsic > 0) & np.isfinite(R_intrinsic)
    return q_sf[mask], R_intrinsic[mask], dR_intrinsic[mask], dq_sf[mask]


def fit_rfxsf(
    sf_file: str,
    save_plot: str | None = None,
    show_plot: bool = True,
    verbose: bool = True,
    de_maxiter: int = 150,
) -> Dict[str, Any]:
    if verbose:
        print("=" * 60)
        print("RFXSF (Chen Shen methodology)")
        print("=" * 60)

    data_sf = np.loadtxt(sf_file, skiprows=29)
    q_sf = data_sf[:, 0]
    SF = data_sf[:, 1]
    dSF = data_sf[:, 2]
    dq_sf = 2.0 * data_sf[:, 3]
    sigma_R = data_sf[:, 4]

    q, R_intrinsic, dR_intrinsic, dq = _create_intrinsic_data(q_sf, SF, dSF, dq_sf)

    if verbose:
        print(f"File: {sf_file}")
        print(f"Using {len(q)} of {len(q_sf)} data points")
        print(f"q range: {q.min():.3f} to {q.max():.3f} Å⁻¹")
        print(f"Thermal roughness: {np.mean(sigma_R):.2f} Å")

    ds = ReflectDataset(data=(q, R_intrinsic, dR_intrinsic))
    ds.x_err = dq

    model, structure = create_model()

    # Estimate scale from mid-q
    R_model_test = model(q)
    q_mid = (q >= 0.08) & (q <= 0.3)
    if q_mid.sum() > 5:
        scale_est = np.median(R_intrinsic[q_mid] / np.maximum(R_model_test[q_mid], 1e-30))
        scale_est = float(np.clip(scale_est, 0.01, 100))
    else:
        scale_est = 1.0
    model.scale.value = scale_est
    model.scale.setp(vary=True, bounds=(scale_est * 0.1, scale_est * 10))

    # No background for intrinsic structure
    model.bkg.setp(vary=False)

    obj = Objective(model, ds)
    chi2 = _fit_objective(obj, verbose=verbose, maxiter=de_maxiter)

    npts = len(q)
    nvary = len(obj.varying_parameters())
    sigma_R_avg = float(np.mean(sigma_R[: len(q)]))

    tails_layer = structure[1]
    heads_layer = structure[2]

    results = {
        "method": "RFXSF",
        "file": sf_file,
        "total_thickness": tails_layer.thick.value + heads_layer.thick.value,
        "tails_thick": tails_layer.thick.value,
        "tails_sld": tails_layer.sld.real.value,
        "heads_thick": heads_layer.thick.value,
        "heads_sld": heads_layer.sld.real.value,
        "heads_vfsolv": heads_layer.vfsolv.value,
        "scale": model.scale.value,
        "bkg": getattr(model, "bkg", 0.0),
        "chi2_red": chi2 / max(1, (npts - nvary)),
        "sigma_R": sigma_R_avg,
        "npts": npts,
        "q_data": q,
        "R_data": R_intrinsic,
        "dR_data": dR_intrinsic,
        "R_fit": model(q),
    }

    # Verbose and plotting are suppressed/unused in process_gixos pipeline.
    return results


