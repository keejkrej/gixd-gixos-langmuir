import numpy as np
import xarray as xr
from pathlib import Path

# Mapping for file suffixes in the raw dataset
KIND_SUFFIX = {
    "sf": "SF",
    "r": "R",
    "ds2rrf": "DS2RRF",
}


def load_gixos_xarray(data_path, name: str, index: int, kind: str = "sf", skiprows: int = 30):
    """
    Load a 1D GIXOS curve as an xarray.DataArray.

    Parameters:
    - data_path: base directory containing sample subfolders
    - name: sample name (subfolder and filename prefix)
    - index: integer index embedded in the filename (zero-padded to 5 in raw data)
    - kind: one of {"sf", "r", "ds2rrf"}
    - skiprows: header lines to skip in raw .dat files

    Returns: DataArray with dims ("qz",) and coord "qz"
    """
    k = kind.lower()
    if k not in KIND_SUFFIX:
        raise ValueError(f"Unknown kind '{kind}'. Must be one of {list(KIND_SUFFIX)}")

    data_dir = Path(data_path) / name
    suffix = KIND_SUFFIX[k]
    filename = f"{name}_{index:05d}_{suffix}.dat"
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(str(path))

    raw = np.loadtxt(path, skiprows=skiprows)

    # Column mapping by kind
    # SF: qz, SF, dSF, dqz, sigma_R, exp(-qz2sigma2)
    # R:  qz, R, dR,  dqz
    # DS2RRF: qz, DS/(R/RF)
    if k == "sf":
        qz = raw[:, 0]
        y = raw[:, 1]
        name_da = "sf"
    elif k == "r":
        qz = raw[:, 0]
        y = raw[:, 1]
        name_da = "r"
    else:  # ds2rrf
        qz = raw[:, 0]
        y = raw[:, 1]
        name_da = "ds2rrf"

    return xr.DataArray(y, dims=["qz"], coords={"qz": qz}, name=name_da)


def parse_index_pressure_from_filename(filename: str):
    """
    Parse measurement index and pressure from a processed filename pattern:
      ..._{index}_{pressure}_{kind}.nc
    Returns (index:int, pressure:float, kind:str)
    """
    import re

    m = re.search(r"_(\d+)_([\d.]+|NA)_(sf|r|ds2rrf)\.nc$", filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse index/pressure/kind from: {filename}")
    idx = int(m.group(1))
    pres = m.group(2)
    pressure = float(pres) if pres != "NA" else float("nan")
    kind = m.group(3).lower()
    return idx, pressure, kind

