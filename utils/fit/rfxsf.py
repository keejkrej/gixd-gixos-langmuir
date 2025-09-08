import numpy as np
import matplotlib.pyplot as plt
from refnx.dataset import ReflectDataset
from refnx.reflect import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter


def calculate_fresnel_rf(q, rho_water=9.43e-6):
    k = q / 2.0
    kz_air = k
    kz_water = np.sqrt(k**2 - 4 * np.pi * rho_water + 0j)
    r_fresnel = (kz_air - kz_water) / (kz_air + kz_water)
    RF = np.abs(r_fresnel) ** 2
    return RF


def create_intrinsic_data(q_sf, SF, dSF, dq_sf):
    RF = calculate_fresnel_rf(q_sf)
    R_intrinsic = RF * SF
    dR_intrinsic = RF * dSF
    mask = (q_sf >= 0.02) & (q_sf <= 0.8) & (R_intrinsic > 0) & np.isfinite(R_intrinsic)
    return q_sf[mask], R_intrinsic[mask], dR_intrinsic[mask], dq_sf[mask]


def create_model_chen_methodology():
    air = SLD(0.0, name="air")
    water = SLD(9.43, name="water")
    tails = SLD(8.0, name="tails")
    heads = SLD(13.0, name="heads")
    structure = air | tails(18, 2.0) | heads(8, 2.0, vfsolv=0.2) | water
    for i in range(len(structure)):
        structure[i].rough.setp(vary=True, bounds=(0.5, 5.0))
    model = ReflectModel(structure, bkg=0.0, scale=1.0)
    tails_layer = structure[1]
    heads_layer = structure[2]
    # Loosen tail thickness bounds to allow broader variation
    tails_layer.thick.setp(vary=True, bounds=(10, 30))
    # Narrow SLD bounds to effectively fix SLDs (reduce degeneracy)
    tails_layer.sld.real.setp(vary=True, bounds=(7.95, 8.05))
    # Narrow head thickness bounds (effectively fixed around 8 Å)
    heads_layer.thick.setp(vary=True, bounds=(7.95, 8.05))
    heads_layer.sld.real.setp(vary=True, bounds=(12.95, 13.05))
    # Loosen headgroup volume fraction bounds
    heads_layer.vfsolv.setp(vary=True, bounds=(0.0, 0.8))
    return model, structure


def fit_intrinsic_structure(obj, verbose=True, maxiter: int = 150):
    fitter = CurveFitter(obj)
    if verbose:
        print("Fitting intrinsic structure (RF×SF)...")
        print("Stage 1: Global search (Differential Evolution)")
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
        print("Stage 2: Local refinement (Least Squares)")
    fitter.fit("least_squares")
    return obj.chisqr()


def plot_chen_methodology(
    q_fit,
    R_intrinsic,
    dR_intrinsic,
    model,
    structure,
    chi2_final,
    npts,
    nvary,
    sigma_R_avg,
    save_path=None,
    show=True,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    R_fit = model(q_fit)
    chi2_red = chi2_final / (npts - nvary)
    axes[0, 0].errorbar(
        q_fit,
        R_intrinsic,
        yerr=dR_intrinsic,
        fmt="o",
        ms=3,
        alpha=0.7,
        label="RF×SF (intrinsic)",
    )
    axes[0, 0].plot(q_fit, R_fit, "-", color="red", lw=2, label="Fitted model")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("q (Å⁻¹)")
    axes[0, 0].set_ylabel("RF×SF")
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Intrinsic Fit (χ²ᵣ={chi2_red:.2f})")
    axes[0, 0].grid(True, alpha=0.3)
    residuals = (R_intrinsic - R_fit) / dR_intrinsic
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


def print_chen_results(obj, model, structure, chi2_final, npts, nvary, sigma_R_avg):
    chi2_red = chi2_final / (npts - nvary)
    print("FITTED PARAMETERS - Intrinsic (RF×SF)")
    print("-" * 50)
    for p in obj.varying_parameters():
        stderr = p.stderr if p.stderr is not None else 0
        print(f"{p.name:20s} = {p.value:8.3f} ± {stderr:8.3f}")
    tails_layer = structure[1]
    head_layer = structure[2]
    total_thickness = tails_layer.thick.value + head_layer.thick.value
    print("\nMONOLAYER STRUCTURE (Intrinsic)")
    print("-" * 50)
    print(f"Total thickness:    {total_thickness:.1f} Å")
    print(
        f"Tails (alkyl):      {tails_layer.thick.value:.1f} Å, SLD = {tails_layer.sld.real.value:.2f}"
    )
    print(
        f"Head (azobenzene):  {head_layer.thick.value:.1f} Å, SLD = {head_layer.sld.real.value:.2f}"
    )
    avg_intrinsic_rough = np.mean(
        [layer.rough.value for layer in structure if hasattr(layer.rough, "value")]
    )
    print(f"Intrinsic roughness: {avg_intrinsic_rough:.2f} Å (interfacial width)")
    print(f"Thermal roughness:   {sigma_R_avg:.2f} Å (capillary waves)")
    print(f"Scale factor:        {model.scale.value:.3f}")
    print("\nFIT QUALITY")
    print("-" * 50)
    if chi2_red < 2:
        print("✅ EXCELLENT intrinsic structure determination")
    elif chi2_red < 5:
        print("✅ GOOD intrinsic structure determination")
    elif chi2_red < 15:
        print("⚠️ ACCEPTABLE - some systematic deviations")
    else:
        print("❌ POOR fit - check model or data")
    print(
        "\nNOTE: This represents the intrinsic chemical structure\nwithout thermal roughness broadening (capillary waves)."
    )


def load_data_from_file(sf_file):
    data_sf = np.loadtxt(sf_file, skiprows=29)
    q_sf = data_sf[:, 0]
    SF = data_sf[:, 1]
    dSF = data_sf[:, 2]
    dq_sf = 2.0 * data_sf[:, 3]
    sigma_R = data_sf[:, 4]
    return q_sf, SF, dSF, dq_sf, sigma_R


def fit_rfxsf(
    sf_file, save_plot=None, show_plot=True, verbose=True, de_maxiter: int = 150
):
    if verbose:
        print("=" * 60)
        print("RFXSF (Chen Shen methodology)")
        print("=" * 60)
    q_sf, SF, dSF, dq_sf, sigma_R = load_data_from_file(sf_file)
    q_fit, R_intrinsic, dR_intrinsic, dq_fit = create_intrinsic_data(
        q_sf, SF, dSF, dq_sf
    )
    if verbose:
        print(f"File: {sf_file}")
        print(f"Using {len(q_fit)} of {len(q_sf)} data points")
        print(f"q range: {q_fit.min():.3f} to {q_fit.max():.3f} Å⁻¹")
        print(f"Thermal roughness: {np.mean(sigma_R):.2f} Å")
    ds = ReflectDataset(data=(q_fit, R_intrinsic, dR_intrinsic))
    ds.x_err = dq_fit
    model, structure = create_model_chen_methodology()
    R_model_test = model(q_fit)
    q_mid = (q_fit >= 0.08) & (q_fit <= 0.3)
    if q_mid.sum() > 5:
        scale_est = np.median(
            R_intrinsic[q_mid] / np.maximum(R_model_test[q_mid], 1e-30)
        )
        scale_est = np.clip(scale_est, 0.01, 100)
    else:
        scale_est = 1.0
    model.scale.value = scale_est
    model.scale.setp(vary=True, bounds=(scale_est * 0.1, scale_est * 10))
    model.bkg.setp(vary=False)
    obj = Objective(model, ds)
    chi2_final = fit_intrinsic_structure(obj, verbose=verbose, maxiter=de_maxiter)
    npts = len(q_fit)
    nvary = len(obj.varying_parameters())
    sigma_R_avg = np.mean(sigma_R[: len(q_fit)])
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
        "chi2_red": chi2_final / (npts - nvary),
        "sigma_R": sigma_R_avg,
        "npts": npts,
        "q_data": q_fit,
        "R_data": R_intrinsic,
        "dR_data": dR_intrinsic,
        "R_fit": model(q_fit),
    }
    if verbose:
        print_chen_results(obj, model, structure, chi2_final, npts, nvary, sigma_R_avg)
    if save_plot or show_plot:
        plot_chen_methodology(
            q_fit,
            R_intrinsic,
            dR_intrinsic,
            model,
            structure,
            chi2_final,
            npts,
            nvary,
            sigma_R_avg,
            save_path=save_plot,
            show=show_plot,
        )
    return results
