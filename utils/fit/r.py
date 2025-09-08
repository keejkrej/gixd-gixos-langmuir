import numpy as np
import matplotlib.pyplot as plt
from refnx.dataset import ReflectDataset
from refnx.reflect import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter


def clean_data(q_r, R_r, dR_r, dqz_r):
    mask_physical = (R_r > 0) & np.isfinite(R_r) & np.isfinite(dR_r) & (dR_r > 0)
    mask_qrange = (q_r >= 0.025) & (q_r <= 0.6)
    mask_clean = mask_physical & mask_qrange
    for i in range(1, len(R_r) - 1):
        if R_r[i] > 100 * max(R_r[i - 1], R_r[i + 1]) and R_r[i] > 1000:
            mask_clean[i] = False
    return q_r[mask_clean], R_r[mask_clean], dR_r[mask_clean], 2.0 * dqz_r[mask_clean]


def create_model():
    air = SLD(0.0, name="air")
    water = SLD(9.41, name="water")
    tails = SLD(8.0, name="tails")
    azo_head = SLD(13.0, name="azo_head")
    structure = air | tails(18, 2.5) | azo_head(8, 2.5, vfsolv=0.2) | water
    for i in range(len(structure)):
        structure[i].rough.setp(vary=True, bounds=(0.5, 6.0))
    model = ReflectModel(structure, bkg=0.0, scale=1.0)
    tails_layer = structure[1]
    head_layer = structure[2]
    # Loosen tail thickness bounds to allow broader variation
    tails_layer.thick.setp(vary=True, bounds=(10, 30))
    # Narrow SLD bounds to effectively fix SLDs
    tails_layer.sld.real.setp(vary=True, bounds=(7.95, 8.05))
    # Narrow head thickness bounds (effectively fixed around 8 Å)
    head_layer.thick.setp(vary=True, bounds=(7.95, 8.05))
    head_layer.sld.real.setp(vary=True, bounds=(12.95, 13.05))
    # Loosen headgroup volume fraction bounds
    head_layer.vfsolv.setp(vary=True, bounds=(0.0, 0.8))
    return model, structure


def estimate_scale(model, q_fit, R_fit):
    R_model_test = model(q_fit)
    q_mid_mask = (q_fit >= 0.08) & (q_fit <= 0.2)
    if q_mid_mask.sum() > 3:
        scale_est = np.median(
            R_fit[q_mid_mask] / np.maximum(R_model_test[q_mid_mask], 1e-30)
        )
        return np.clip(scale_est, 0.01, 1000)
    return 1.0


def fit_model(obj, verbose=True, maxiter: int = 200):
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


def plot_results(
    q_fit,
    R_fit,
    dR_fit,
    model,
    chi2_final,
    npts,
    nvary,
    structure,
    save_path=None,
    show=True,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    R_final = model(q_fit)
    chi2_red = chi2_final / (npts - nvary)
    axes[0, 0].errorbar(
        q_fit, R_fit, yerr=dR_fit, fmt="o", ms=3, alpha=0.7, label="Data"
    )
    axes[0, 0].plot(q_fit, R_final, "-", color="red", lw=2, label="Best fit")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("q (Å⁻¹)")
    axes[0, 0].set_ylabel("R")
    axes[0, 0].legend()
    title_color = "green" if chi2_red < 5 else "orange" if chi2_red < 20 else "red"
    axes[0, 0].set_title(f"Final Fit (χ²ᵣ={chi2_red:.2f})", color=title_color)
    axes[0, 0].grid(True, alpha=0.3)
    residuals = (R_fit - R_final) / dR_fit
    axes[0, 1].plot(q_fit, residuals, "o", ms=4)
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].axhline(2, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].axhline(-2, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_xlabel("q (Å⁻¹)")
    axes[0, 1].set_ylabel("(Data - Model) / σ")
    axes[0, 1].set_title("Normalized Residuals")
    axes[0, 1].grid(True, alpha=0.3)
    z, sld = structure.sld_profile()
    axes[1, 0].plot(z, sld, "b-", lw=2)
    axes[1, 0].set_xlabel("z (Å)")
    axes[1, 0].set_ylabel("SLD (×10⁻⁶ Å⁻²)")
    axes[1, 0].set_title("SLD Profile")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].text(
        0.5,
        0.5,
        "Parameter\nSummary",
        ha="center",
        va="center",
        transform=axes[1, 1].transAxes,
        fontsize=12,
    )
    axes[1, 1].set_title("Results Summary")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def print_results(obj, model, structure, chi2_final, npts, nvary):
    chi2_red = chi2_final / (npts - nvary)
    print("FITTED PARAMETERS")
    print("-" * 50)
    for p in obj.varying_parameters():
        stderr = p.stderr if p.stderr is not None else 0
        print(f"{p.name:20s} = {p.value:8.3f} ± {stderr:8.3f}")
    tails_layer = structure[1]
    head_layer = structure[2]
    total_thickness = tails_layer.thick.value + head_layer.thick.value
    print("\nMONOLAYER STRUCTURE")
    print("-" * 50)
    print(f"Total thickness:    {total_thickness:.1f} Å")
    print(
        f"Tails (alkyl):      {tails_layer.thick.value:.1f} Å, SLD = {tails_layer.sld.real.value:.2f}"
    )
    print(
        f"Head (azobenzene):  {head_layer.thick.value:.1f} Å, SLD = {head_layer.sld.real.value:.2f}"
    )
    avg_roughness = np.mean(
        [layer.rough.value for layer in structure if hasattr(layer.rough, "value")]
    )
    print(f"Average roughness:  {avg_roughness:.2f} Å")
    print(f"Scale factor:       {model.scale.value:.3f}")
    print("\nFIT QUALITY")
    print("-" * 50)
    if chi2_red < 2:
        print("✅ EXCELLENT fit quality")
    elif chi2_red < 5:
        print("✅ GOOD fit quality")
    elif chi2_red < 15:
        print("⚠️ ACCEPTABLE fit - some systematic deviations")
    else:
        print("❌ POOR fit - model inadequate or data issues")
    print(f"Reduced χ² = {chi2_red:.2f}")
    print(f"Data points: {npts}, Parameters: {nvary}")


def fit_r(r_file, save_plot=None, show_plot=True, verbose=True, de_maxiter: int = 200):
    if verbose:
        print("=" * 60)
        print("R DATA FITTING")
        print("=" * 60)
    q_r, R_r, dR_r, dqz_r = np.loadtxt(r_file, skiprows=28, unpack=True)
    q_fit, R_fit, dR_fit, dq_fit = clean_data(q_r, R_r, dR_r, dqz_r)
    if verbose:
        print(f"File: {r_file}")
        print(f"Using {len(q_fit)} of {len(q_r)} data points")
        print(f"q range: {q_fit.min():.3f} to {q_fit.max():.3f} Å⁻¹")
        print(f"R range: {R_fit.min():.2e} to {R_fit.max():.2e}")
    ds = ReflectDataset(data=(q_fit, R_fit, dR_fit))
    ds.x_err = dq_fit
    model, structure = create_model()
    scale_est = estimate_scale(model, q_fit, R_fit)
    model.scale.value = scale_est
    model.scale.setp(vary=True, bounds=(scale_est * 0.1, scale_est * 10))
    model.bkg.setp(vary=True, bounds=(0, np.min(R_fit) * 0.5))
    obj = Objective(model, ds)
    chi2_final = fit_model(obj, verbose=verbose, maxiter=de_maxiter)
    npts = len(q_fit)
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
        "chi2_red": chi2_final / (npts - nvary),
        "npts": npts,
        "q_data": q_fit,
        "R_data": R_fit,
        "dR_data": dR_fit,
        "R_fit": model(q_fit),
    }
    if verbose:
        print_results(obj, model, structure, chi2_final, npts, nvary)
    if save_plot or show_plot:
        plot_results(
            q_fit,
            R_fit,
            dR_fit,
            model,
            chi2_final,
            npts,
            nvary,
            structure,
            save_path=save_plot,
            show=show_plot,
        )
    return results
