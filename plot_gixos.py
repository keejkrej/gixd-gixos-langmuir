from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROCESSED_DIR = "processed/gixos"
PLOT_DIR = "plot/gixos"


def create_pressure_plots(df: pd.DataFrame | None, output_dir: Path):
    if df is None or df.empty:
        print("No per-measurement CSVs found; skipping pressure plots.")
        return

    df_sorted = df.sort_values(["Name", "Pressure_mN_per_m"]).copy()

    # Build a consistent color map per sample
    sample_names = list(df_sorted["Name"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(sample_names)}

    def make_fig(method: str, tails_col: str, vf_col: str, chi_col: str, fname: str):
        # Filter by chi-squared threshold
        dff = df_sorted.copy()
        if chi_col in dff.columns:
            dff = dff[dff[chi_col] <= 50]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Tails thickness vs pressure
        ax = axes[0]
        for name, grp in dff.groupby("Name"):
            color = colors[name]
            pressures = grp["Pressure_mN_per_m"].values
            if tails_col in grp:
                ax.plot(pressures, grp[tails_col].values, "-o", color=color, linewidth=2, markersize=5, label=name)
        ax.set_xlabel("Surface Pressure (mN/m)")
        ax.set_ylabel("Tails Thickness (Å)")
        ax.set_title(f"{method}: Tails Thickness vs Pressure")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

        # Head group fraction (1 - vfsolv) vs pressure
        ax = axes[1]
        for name, grp in dff.groupby("Name"):
            color = colors[name]
            pressures = grp["Pressure_mN_per_m"].values
            if vf_col in grp:
                head_group_frac = 1.0 - grp[vf_col].values
                ax.plot(pressures, head_group_frac, "-o", color=color, linewidth=2, markersize=5, label=name)
        ax.set_xlabel("Surface Pressure (mN/m)")
        ax.set_ylabel("Head Group Fraction (1 - vfsolv)")
        ax.set_title(f"{method}: Head Group Fraction vs Pressure")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = output_dir / fname
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

    # Separate figures for R and RFXSF with only tails and (1 - vfsolv)
    make_fig("R", "R_Tails_A", "R_Head_VF", "R_Chi2", "pressure_analysis_R.png")
    make_fig("RFXSF", "RFXSF_Tails_A", "RFXSF_Head_VF", "RFXSF_Chi2", "pressure_analysis_RFXSF.png")


def _collect_nc_pairs(processed_dir: Path):
    pairs = {}
    for sample_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        sample = sample_dir.name
        for nc_path in sorted(sample_dir.glob(f"{sample}_*_*.nc")):
            m = re.search(rf"{re.escape(sample)}_(\d+)_([\-\d\.]+)_(rfxsf|r)\.nc$", nc_path.name)
            if not m:
                continue
            idx = int(m.group(1))
            try:
                pressure = float(m.group(2))
            except ValueError:
                pressure = np.nan
            method = m.group(3).lower()
            key = (sample, idx, pressure)
            pairs.setdefault(key, {})[method] = nc_path
    return pairs


def _plot_single(ax, ds: xr.Dataset, title: str, ylabel: str):
    q = ds["q"].values
    r = ds["R_data"].values
    dr = ds["dR_data"].values
    rfit = ds["R_fit"].values
    ax.errorbar(q, r, yerr=dr, fmt="o", ms=3, alpha=0.7, label="data")
    ax.plot(q, rfit, "-", lw=2, label="fit")
    ax.set_yscale("log")
    ax.set_xlabel("q (Å⁻¹)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)


def create_data_fit_plots(processed_dir: Path, plot_dir: Path):
    pairs = _collect_nc_pairs(processed_dir)
    if not pairs:
        print("No NetCDF fit files found; skipping data-fit plots.")
        return

    for (sample, idx, pressure), methods in pairs.items():
        out_dir = plot_dir / sample
        out_dir.mkdir(parents=True, exist_ok=True)

        r_path = methods.get("r")
        rfxsf_path = methods.get("rfxsf")

        if r_path and rfxsf_path:
            ds_r = xr.open_dataset(r_path)
            ds_rx = xr.open_dataset(rfxsf_path)
            chi_r = ds_r.attrs.get("chi2_red", np.nan)
            chi_rx = ds_rx.attrs.get("chi2_red", np.nan)
            t_r = ds_r.attrs.get("total_thickness_A", np.nan)
            t_rx = ds_rx.attrs.get("total_thickness_A", np.nan)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            _plot_single(
                axes[0],
                ds_rx,
                f"{sample}_{idx} p={pressure} mN/m\nRFXSF: t={t_rx:.1f} Å, χ²ᵣ={chi_rx:.2f}",
                "RF×SF (intrinsic)",
            )
            _plot_single(
                axes[1],
                ds_r,
                f"{sample}_{idx} p={pressure} mN/m\nR: t={t_r:.1f} Å, χ²ᵣ={chi_r:.2f}",
                "Reflectivity",
            )
            plt.tight_layout()
            out_path = out_dir / f"{sample}_{idx}_{pressure}_comparison.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")
        else:
            # Plot whichever method exists
            method = "rfxsf" if rfxsf_path else "r"
            ds = xr.open_dataset(rfxsf_path or r_path)
            chi = ds.attrs.get("chi2_red", np.nan)
            t = ds.attrs.get("total_thickness_A", np.nan)
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            _plot_single(
                ax,
                ds,
                f"{sample}_{idx} p={pressure} mN/m\n{method.upper()}: t={t:.1f} Å, χ²ᵣ={chi:.2f}",
                "RF×SF (intrinsic)" if method == "rfxsf" else "Reflectivity",
            )
            plt.tight_layout()
            out_path = out_dir / f"{sample}_{idx}_{pressure}_{method}_fit.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")

    # Done creating data-fit plots


def create_sample_overlays(processed_dir: Path, plot_dir: Path):
    pairs = _collect_nc_pairs(processed_dir)
    if not pairs:
        print("No NetCDF fit files found; skipping overlay plots.")
        return

    # Group files by sample then by method
    by_sample = {}
    for (sample, idx, pressure), methods in pairs.items():
        by_sample.setdefault(sample, []).append((idx, pressure, methods))

    for sample, entries in by_sample.items():
        out_dir = plot_dir / sample
        out_dir.mkdir(parents=True, exist_ok=True)

        for method in ("r", "rfxsf"):
            # Collect list of (pressure, idx, path)
            items = []
            for idx, pressure, methods in entries:
                path = methods.get(method)
                if path is not None:
                    items.append((pressure, idx, path))
            if not items:
                continue
            # Sort by pressure then idx
            items.sort(key=lambda t: (np.inf if np.isnan(t[0]) else t[0], t[1]))

            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            cmap = plt.get_cmap("viridis")
            for i, (pressure, idx, path) in enumerate(items):
                ds = xr.open_dataset(path)
                q = ds["q"].values
                r = ds["R_data"].values
                dr = ds["dR_data"].values
                rfit = ds["R_fit"].values
                color = cmap(i / max(1, len(items) - 1))
                # Data points
                ax.errorbar(
                    q, r, yerr=dr, fmt="o", ms=2.5, alpha=0.5, color=color, label=None
                )
                # Fit line
                ax.plot(q, rfit, "-", lw=2, color=color, label=f"p={pressure} mN/m")

            ax.set_yscale("log")
            ax.set_xlabel("qz (Å⁻¹)")
            ax.set_ylabel("RF×SF (intrinsic)" if method == "rfxsf" else "Reflectivity")
            ax.set_title(f"{sample} - Overlay ({method.upper()})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            out_path = out_dir / f"{sample}_overlay_{method}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


def create_method_sample_grid(processed_dir: Path, plot_dir: Path):
    pairs = _collect_nc_pairs(processed_dir)
    if not pairs:
        print("No NetCDF fit files found; skipping method-sample grid plot.")
        return

    # Collect samples and group entries by (sample, method)
    sample_names = sorted({s for (s, _, _ ) in pairs.keys()})
    methods = ["r", "rfxsf"]
    nrows = len(methods)
    ncols = len(sample_names)
    if ncols == 0:
        return
    figsize = (4 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # For each sample and method, overlay all pressures
    for col, sample in enumerate(sample_names):
        for row, method in enumerate(methods):
            ax = axes[row, col]
            # Collect all entries for this sample/method
            items = []
            for (s, idx, pressure), m in pairs.items():
                if s == sample and method in m:
                    items.append((pressure, idx, m[method]))
            if not items:
                ax.set_axis_off()
                continue
            items.sort(key=lambda t: (np.inf if np.isnan(t[0]) else t[0], t[1]))

            cmap = plt.get_cmap("viridis")
            for i, (pressure, idx, path) in enumerate(items):
                ds = xr.open_dataset(path)
                q = ds["q"].values
                r = ds["R_data"].values
                dr = ds["dR_data"].values
                rfit = ds["R_fit"].values
                color = cmap(i / max(1, len(items) - 1))
                chi = ds.attrs.get("chi2_red", np.nan)
                # data
                ax.errorbar(q, r, yerr=dr, fmt="o", ms=2.5, alpha=0.5, color=color)
                # fit with legend label including R^2 and thickness
                t_tot = ds.attrs.get("total_thickness_A", np.nan)
                ax.plot(q, rfit, "-", lw=2, color=color,
                        label=f"p={pressure} mN/m, t={t_tot:.1f} Å, χ²ᵣ={chi:.3f}")

            ax.set_yscale("log")
            ax.set_xlabel("qz (Å⁻¹)")
            ax.set_ylabel("RF×SF" if method == "rfxsf" else "Reflectivity")
            title_method = method.upper()
            ax.set_title(f"{sample} ({title_method})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    out = plot_dir / "data_fit_grid.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def create_sample_method_panels(processed_dir: Path, plot_dir: Path):
    pairs = _collect_nc_pairs(processed_dir)
    if not pairs:
        print("No NetCDF fit files found; skipping per-sample panels.")
        return

    # Group by sample and method
    by_sample = {}
    for (sample, idx, pressure), methods in pairs.items():
        for method, path in methods.items():
            by_sample.setdefault((sample, method), []).append((pressure, idx, path))

    for (sample, method), items in by_sample.items():
        # Sort by pressure then idx
        items.sort(key=lambda t: (np.inf if np.isnan(t[0]) else t[0], t[1]))
        n = len(items)
        if n == 0:
            continue
        # Choose a compact grid
        ncols = 2 if n <= 6 else 3
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        ax_list = axes.ravel()
        for ax in ax_list[n:]:
            ax.set_axis_off()

        for k, (pressure, idx, path) in enumerate(items):
            ds = xr.open_dataset(path)
            q = ds["q"].values
            r = ds["R_data"].values
            dr = ds["dR_data"].values
            rfit = ds["R_fit"].values
            chi = ds.attrs.get("chi2_red", np.nan)
            ax = ax_list[k]
            # Data and fit
            ax.errorbar(q, r, yerr=dr, fmt="o", ms=2.5, alpha=0.6, color="C0")
            ax.plot(q, rfit, "-", lw=2, color="C1")
            ax.set_yscale("log")
            ax.set_xlabel("qz (Å⁻¹)")
            ax.set_ylabel("RF×SF" if method == "rfxsf" else "Reflectivity")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"p={pressure} mN/m (idx {idx})")
            # Annotation box from per-measurement CSV (if available), otherwise attrs
            csv_path = Path(str(path).replace(".nc", ".csv"))
            if csv_path.exists():
                import pandas as _pd
                row = _pd.read_csv(csv_path).iloc[0].to_dict()
                exclude = {"sample", "method", "index", "pressure_mN_per_m", "file"}
                lines = []
                for k2, v in row.items():
                    if k2 in exclude:
                        continue
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        lines.append(f"{k2}={v:.3f}")
                    else:
                        lines.append(f"{k2}={v}")
                txt = "\n".join(lines)
            else:
                t_tot = ds.attrs.get("total_thickness_A", np.nan)
                txt = f"chi2_red={chi:.3f}\n tails_thick_A={ds.attrs.get('tails_thick_A', np.nan):.3f}\n heads_vfsolv={ds.attrs.get('heads_vfsolv', np.nan):.3f}"
            ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8))

        plt.suptitle(f"{sample} — {method.upper()} data & fit", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_dir = plot_dir / sample
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sample}_{method}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


# Removed summary plots; all parameters are consumed from per-measurement CSVs


def create_pressure_plots2(df: pd.DataFrame, output_dir: Path):
    df_sorted = df.sort_values(["Name", "Pressure_mN_per_m"]).copy()

    # Build a consistent color map per sample
    sample_names = list(df_sorted["Name"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(sample_names)}

    processed_dir = Path(PROCESSED_DIR)

    def ensure_head_vf_column(prefix: str, dfref: pd.DataFrame):
        col = f"{prefix}_Head_VF"
        if col in dfref.columns:
            return col
        vals = []
        for _, row in dfref.iterrows():
            name = row["Name"]
            idx = int(row["Index"])
            p = row["Pressure_mN_per_m"]
            method = "r" if prefix == "R" else "rfxsf"
            csv_path = processed_dir / name / f"{name}_{idx}_{p}_{method}.csv"
            if csv_path.exists():
                try:
                    sub = pd.read_csv(csv_path)
                    vals.append(float(sub.loc[0, "heads_vfsolv"]))
                except Exception:
                    vals.append(np.nan)
            else:
                vals.append(np.nan)
        dfref[col] = vals
        return col

    r_vf_col = ensure_head_vf_column("R", df_sorted)
    rfxsf_vf_col = ensure_head_vf_column("RFXSF", df_sorted)

    def make_fig(method: str, tails_col: str, vf_col: str, chi_col: str, fname: str):
        dff = df_sorted.copy()
        if chi_col in dff.columns:
            dff = dff[dff[chi_col] <= 50]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Tails thickness vs pressure
        ax = axes[0]
        for name, grp in dff.groupby("Name"):
            color = colors[name]
            pressures = grp["Pressure_mN_per_m"].values
            if tails_col in grp:
                ax.plot(pressures, grp[tails_col].values, "-o", color=color, linewidth=2, markersize=5, label=name)
        ax.set_xlabel("Surface Pressure (mN/m)")
        ax.set_ylabel("Tails Thickness (Å)")
        ax.set_title(f"{method}: Tails Thickness vs Pressure")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

        # Head group fraction (1 - vfsolv) vs pressure
        ax = axes[1]
        for name, grp in dff.groupby("Name"):
            color = colors[name]
            pressures = grp["Pressure_mN_per_m"].values
            if vf_col in grp:
                head_group_frac = 1.0 - grp[vf_col].values
                ax.plot(pressures, head_group_frac, "-o", color=color, linewidth=2, markersize=5, label=name)
        ax.set_xlabel("Surface Pressure (mN/m)")
        ax.set_ylabel("Head Group Fraction (1 - vfsolv)")
        ax.set_title(f"{method}: Head Group Fraction vs Pressure")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = output_dir / fname
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

    make_fig("R", "R_Tails_A", r_vf_col, "R_Chi2", "pressure_analysis_R.png")
    make_fig("RFXSF", "RFXSF_Tails_A", rfxsf_vf_col, "RFXSF_Chi2", "pressure_analysis_RFXSF.png")


def build_summary_from_measurements(processed_dir: Path) -> pd.DataFrame | None:
    rows = []
    for sample_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        sample = sample_dir.name
        for csv_path in sample_dir.glob(f"{sample}_*_*.csv"):
            m = re.search(rf"{re.escape(sample)}_(\d+)_([\-\d\.]+)_(rfxsf|r)\.csv$", csv_path.name)
            if not m:
                continue
            idx = int(m.group(1))
            try:
                pressure = float(m.group(2))
            except ValueError:
                pressure = np.nan
            method = m.group(3).lower()
            try:
                d = pd.read_csv(csv_path)
            except Exception:
                continue
            if d.empty:
                continue
            rec = d.iloc[0].to_dict()
            rec_normalized = {
                "Name": sample,
                "Index": idx,
                "Pressure_mN_per_m": pressure,
            }
            # Map fields by method
            if method == "r":
                rec_normalized.update(
                    {
                        "R_Tails_A": rec.get("tails_thick_A", np.nan),
                        "R_Chi2": rec.get("chi2_red", np.nan),
                        "R_Head_VF": rec.get("heads_vfsolv", np.nan),
                    }
                )
            else:  # rfxsf
                rec_normalized.update(
                    {
                        "RFXSF_Tails_A": rec.get("tails_thick_A", np.nan),
                        "RFXSF_Chi2": rec.get("chi2_red", np.nan),
                        "RFXSF_Head_VF": rec.get("heads_vfsolv", np.nan),
                    }
                )
            rows.append(rec_normalized)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Aggregate duplicates (R and RFXSF rows for same measurement) by groupby-first
    df = df.groupby(["Name", "Index", "Pressure_mN_per_m"], as_index=False).first()
    return df


def main():
    processed_dir = Path(PROCESSED_DIR)
    plot_dir = Path(PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1) Create per-sample overlay plots (data+fit across pressures), split by method
    create_sample_overlays(processed_dir, plot_dir)
    # 1b) Create per-sample, per-method multi-panel plots (no overlay)
    create_sample_method_panels(processed_dir, plot_dir)

    # 2) Build an aggregate view directly from per-measurement CSVs and make pressure plots
    df = build_summary_from_measurements(processed_dir)
    create_pressure_plots(df, plot_dir)


if __name__ == "__main__":
    main()
