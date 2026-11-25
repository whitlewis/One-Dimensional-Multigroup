"""Loading saved SM/VMG results, shared by every postprocessing script.

A saved run is one .h5 file (see Base.Base.saveResults):

    datastash/<run set>/<sm | vmg | smfailed | vmgfailed>/<runName>_<timestamp>.h5

with the arrays as datasets (fullTensorPhi, temperatureSet, timeSet, spaceGrid, spaceMid, dt,
iterationsPerStep) and every run parameter as a file attribute. Older files (flat sm/ and vmg/
folders, fewer attributes) load the same way; missing values are derived from the arrays.

Each loaded run is a Result:
    result.data    arrays: fullTensorPhi (nTime, groups, nBins), temperatureSet (nBins, nTime), timeSet
    result.params  every attribute plus spaceGrid, spaceMid, freqGrid, dt, dx, maxDt, method, ...
    result.label   short legend label, e.g. "VMG · G=100 · Nt=4000 · Δx=0.02"
"""
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("src", "postprocessing"):
    if os.path.join(_ROOT, _folder) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _folder))

import h5py
import numpy as np

import Base
import MMS

ARRAYS = ("fullTensorPhi", "temperatureSet", "timeSet")
GRID_ARRAYS = ("spaceGrid", "spaceMid", "dt", "iterationsPerStep")


class Result:
    def __init__(self, data, params, path=None):
        self.data = data
        self.params = params
        self.path = path
        self.fileName = os.path.splitext(os.path.basename(path))[0] if path else params.get("runName", "Run")
        self.folder = os.path.basename(os.path.dirname(path)) if path else params.get("fileFolder", "")
        self.method = params["method"]
        self.label = runLabel(params)

    @property
    def timeCount(self):
        """Number of time samples present in every array (a failed run can stop early)."""
        return min(len(self.data["timeSet"]), self.data["fullTensorPhi"].shape[0],
                   self.data["temperatureSet"].shape[1])

    @property
    def time(self):
        return np.asarray(self.data["timeSet"])[: self.timeCount]

    @property
    def x(self):
        return np.asarray(self.params["spaceMid"])

    def cellAt(self, position):
        """Index of the cell whose centre is nearest to `position` (cm)."""
        return int(np.argmin(np.abs(self.x - position)))

    def materialTemperature(self):
        """Material temperature, shape (nBins, nTime)."""
        return self.data["temperatureSet"][:, : self.timeCount]

    def radiationTemperature(self):
        """Radiation temperature (sum_g phi_g / (a c))^(1/4), shape (nBins, nTime)."""
        energy = np.sum(self.data["fullTensorPhi"][: self.timeCount], axis=1) / (Base.Constants.a * Base.Constants.c)
        return (np.maximum(energy, 0.0) ** 0.25).T

    def __repr__(self):
        return f"Result({self.label!r}, {self.fileName!r})"


def runLabel(params):
    """Short legend label: method, groups, time steps and cell width."""
    return f"{params['method']} · G={params['groups']} · Nt={params['nSteps']} · Δx={params['dx']:.3g}"


def _plain(value):
    """h5py attribute -> plain Python value (str, bool, int, float or ndarray)."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def loadResult(path):
    """Load one .h5 result."""
    with h5py.File(path, "r") as f:
        data = {key: f[key][:] for key in ARRAYS}
        params = {key: _plain(value) for key, value in f.attrs.items()}
        for key in GRID_ARRAYS + ("freqGrid",):
            if key in f:
                params[key] = f[key][:]
    folder = os.path.basename(os.path.dirname(path))

    # Values older files may lack, derived from the arrays
    params.setdefault("groups", data["fullTensorPhi"].shape[1])
    params.setdefault("nBins", data["fullTensorPhi"].shape[2])
    params.setdefault("nSteps", len(data["timeSet"]) - 1)
    params.setdefault("dt", np.diff(data["timeSet"]))
    params.setdefault("dx", float(np.mean(np.diff(params["spaceGrid"]))))
    params.setdefault("xMin", float(params["spaceGrid"][0]))
    params.setdefault("xMax", float(params["spaceGrid"][-1]))
    params.setdefault("maxDt", float(np.max(params["dt"])))
    # Older VMG files predate the movingCoordinates attribute; their folder identifies them
    params["movingCoordinates"] = bool(params.get("movingCoordinates", folder.lower().startswith("vmg")))
    params.setdefault("method", "VMG" if params["movingCoordinates"] else "SM")
    params["MMSCompare"] = bool(params.get("MMSCompare", False))
    params.setdefault("mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET)
    params.setdefault("boundaryLeft", "")
    params.setdefault("boundaryRight", "")
    params.setdefault("runLabel", "Run")
    params.setdefault("failed", folder.lower().endswith("failed"))
    return Result(data, params, path)


def loadResults(paths):
    results = []
    for path in paths:
        results.append(loadResult(path))
        print(f"Loaded {results[-1].label}  ({os.path.relpath(path, Base.DATA_DIR)})")
    return results


def findResults(folder=None, mmsOnly=None, includeFailed=False):
    """Every .h5 under datastash/ (or under `folder`, e.g. a run-set name), sorted by path.

    mmsOnly: None = all runs, True = only MMS runs, False = only non-MMS runs.
    """
    root = Base.DATA_DIR if folder is None else os.path.join(Base.DATA_DIR, folder)
    paths = sorted(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))
    if not includeFailed:
        paths = [p for p in paths if not os.path.basename(os.path.dirname(p)).lower().endswith("failed")]
    readable = []
    for path in paths:   # skip files that are still being written or are corrupt
        try:
            with h5py.File(path, "r") as f:
                isMMS = bool(f.attrs.get("MMSCompare", False))
        except OSError as exc:
            print(f"Skipping unreadable file {os.path.relpath(path, Base.DATA_DIR)}: {exc}")
            continue
        if mmsOnly is None or isMMS == mmsOnly:
            readable.append(path)
    return readable


def pickFiles(title="Select HDF5 results"):
    """File dialog (repeat to add files from other folders); returns a list of paths."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    paths = []
    while True:
        paths += filedialog.askopenfilenames(
            title=title,
            initialdir=Base.DATA_DIR,
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")],
        )
        if not paths or messagebox.askquestion("Add files", "Add files from another folder?") != "yes":
            break
    root.destroy()
    return paths


def resultFromGrid(grid, params):
    """Package an in-memory solve (Base.Grid + Parameters) as a Result, without saving it."""
    data = {
        "fullTensorPhi": grid.fullTensorPhiTime,
        "temperatureSet": grid.temperatureSet,
        "timeSet": grid.timeSet,
    }
    attrs = Base.parameterAttributes(params)
    attrs.update(
        spaceGrid=grid.spaceGrid, spaceMid=grid.spaceMid, freqGrid=grid.freqGrid, dt=grid.dt,
        dx=float(grid.dx), maxDt=float(np.max(grid.dt)), groups=params.freqNum,
        method="VMG" if params.movingCoordinates else "SM",
        MMSCompare=bool(getattr(params, "MMSCompare", False)),
        mmsTemperatureOffset=getattr(params, "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET),
        failed=False,
    )
    return Result(data, attrs)
