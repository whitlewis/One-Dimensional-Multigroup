"""Plotting for SM/VMG results: live grids from a solve and results loaded by AnalyzeData."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("src", "postprocessing"):
    if os.path.join(_ROOT, _folder) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _folder))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from cycler import cycler

import Base
from Base import Constants as const
import DiffEqSolve as diffEq
import MMS


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,

    "axes.labelsize": 12,
    "axes.titlesize": 12,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 9,

    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,

    # Ticks point inward
    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # Colorblind-friendly Okabe-Ito-style colors
    "axes.prop_cycle": cycler("color", [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # purple
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#000000",  # black
    ]),
})


def figurePath(category, name):
    """figures/<category>/<name>; see Base.figurePath."""
    return Base.figurePath(category, name)


# Helpful equations for plot ref (from Logic.py)
# Base Planck definiton
def planck(nu, T):  # Planck function (not group integrated or weighted)
    denom = np.expm1(nu/T)  # exp(x)-1 safely
    f = (15.0 * const.a * const.c) / (4.0 * np.pi**5)
    return f * nu**3 / denom

# def movingPlanck(u, T):


# Simpson for integration over group
def simpson(integrand, lo, hi):
    h = (hi - lo) / 3
    out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
    return out

# Group integrated Planck
def groupPlanck(freqGrid, T):
    # Integrate the Planck function over each frequency group to get group-averaged source
    lo = freqGrid[:-1, None]
    hi = freqGrid[1:, None]
    integrand = lambda nu: planck(nu, T)
    bbar = simpson(integrand, lo, hi)
    return bbar


# Plot the scalar flux for a time and space across all groups
def plotPhiStepFreq(grid, params, step, label="Initial Scalar Flux  accross Frequencies φ"):
    freqs = grid.freqGroups
    phi = grid.fullTensorPhi[:, step]  # shape: (nBins,)
    
    print(f'initial temperature for step {step}: {params.radiationTemperature}')  # Print the initial temperature for the specified time step
    plt.semilogx(freqs, planck(freqs, params.radiationTemperature), label="Planckian")  # Plot the Planck function for reference
    plt.semilogx(freqs, groupPlanck(grid.freqGrid, params.radiationTemperature), label="Group-Integrated Planck")  # Plot the group-integrated Planck function for reference
    plt.semilogx(freqs, phi, label=label, linestyle = '--')
    plt.xlabel("Frequency Groups")
    plt.ylabel("Scalar Flux φ")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figurePath("spectra", f"{label}_{timestamp}.pdf")
    # plt.savefig(filename)
    plt.show()


def plotTemperature(grid):
    timeCount = min(
        len(grid.timeSet),
        len(grid.fullTensorPhiTime),
        grid.temperatureSet.shape[1],
    )
    t = grid.timeSet[:timeCount]
    cellIndex = grid.nBins // 2
    EradSet = []
    for timeIndex in range(timeCount):
        Erad = np.sum(grid.fullTensorPhiTime[timeIndex], axis=0)
        EradSet.append(Erad[cellIndex])
    Trad = (np.array(EradSet)/ const.a / const.c)**0.25
    T = grid.temperatureSet[cellIndex, :timeCount]
    plt.plot(t, T, label="Material temperature")
    plt.plot(t, Trad, label="Radiation temperature", linestyle='--')
    plt.xlabel("t (ns)")
    plt.ylabel("Temperature (keV)")
    plt.legend()
    plt.grid(True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figurePath("temperature", f"Temperature_vs_Time_{timestamp}.pdf")
    plt.savefig(filename)
    plt.show()

def getDiffeq(const, tmax):
    minFreq = 1e-4
    maxFreq = 25
    infFreq = 125
    freqNum = 100
    timeMax = tmax
    timeNum = 100000



    inputDictTest = {
        'colorTemperature' : 1.0,
        'initialTemperature': 0.4,
        "radiationTemperature": 0.5,
        "freqGrid": np.append(np.linspace(minFreq, maxFreq, freqNum), infFreq),
        'minFreq' : minFreq,
        'maxFreq' : maxFreq,
        'infFreq' : infFreq,
        'freqNum' : freqNum,
        'timeMax': timeMax,
        'timeNum': timeNum,
        'timeSet': np.geomspace(1e-14, timeMax, timeNum)
    }


    solution = diffEq.SMSolve(inputDictTest)
    t = solution.t
    T = solution.y[-1]
    Tr = (np.sum(solution.y[:-1],axis=0)/ const.a / const.c)**.25
    return t, T, Tr

def _styleAxis(axis):
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)


def _defaultPosition(results, binIndex=None):
    """x (cm) used for single-cell plots: the centre of cell binIndex of the first run (the
    middle cell if None). Every run then uses its own cell nearest that x, so runs with different
    dx are compared at the same place."""
    x = results[0].x
    return float(x[len(x) // 2 if binIndex is None else min(binIndex, len(x) - 1)])


def _exactMMS(params, position, time):
    """Exact MMS (material, radiation) temperatures at one position."""
    sigmaX = params["mmsSigmaX"]
    spatial = np.exp(-(position ** 2) / (2 * sigmaX**2)) / (sigmaX * np.sqrt(2 * np.pi))
    decay = np.exp(-params["mmsTau"] * time / 2)
    offset = params.get("mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET)
    return (MMS.temperatureFromFourthPowerProfile(params["mmsT0"], (1 - decay) * spatial, offset),
            MMS.temperatureFromFourthPowerProfile(params["mmsT0"], (1 + decay) * spatial, offset))


def plotTemperatureLoaded(results, position=None):
    """Material (solid) and radiation (dashed) temperature vs. time in one cell, for every run,
    plus the 0-D reference when the run has reflective walls."""
    if not results:
        return None
    position = _defaultPosition(results) if position is None else position
    fig, axis = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, result in enumerate(results):
        cell = result.cellAt(position)
        color = colors[i % len(colors)]
        axis.plot(result.time, result.materialTemperature()[cell], color=color, label=f"{result.label} material")
        axis.plot(result.time, result.radiationTemperature()[cell], color=color, linestyle="--",
                  label=f"{result.label} radiation")

    reflective = [r for r in results
                  if r.params["boundaryLeft"] == "Reflective" and r.params["boundaryRight"] == "Reflective"]
    if reflective:
        t, T, Tr = getDiffeq(const, max(r.time[-1] for r in reflective))
        axis.plot(t, T, color="0.2", linestyle="--", label="0-D reference material")
        axis.plot(t, Tr, color="0.2", label="0-D reference radiation")

    axis.set_ylim(0, 0.8)
    axis.set_xlabel("t (ns)")
    axis.set_ylabel("Temperature (keV)")
    axis.set_title(f"Temperature at x = {position:.3g} cm")
    _styleAxis(axis)
    axis.legend(frameon=False, loc="best", fontsize="small")
    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(figurePath("temperature", f"TempPlot_{timestamp}.pdf"))
    plt.show()
    return fig, axis


def plotTemperatureTime(grid):
    x = grid.spaceMid  # cell centers
    time = grid.timeSet
    T_time = grid.temperatureSet  # shape: (nBins, nSteps+1)

    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(
        x,
        time,
        T_time.T,
        shading='auto',
        cmap='viridis'
    )

    fig.colorbar(pcm, label='Temperature T', ax=ax)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Time')
    ax.set_title('Space-Time Temperature Distribution')
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)

    plt.tight_layout()
    plt.show()

def plotTemperatureTimeLoaded(results):
    """Space-time map of the material temperature of the first run."""
    result = results[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(result.x, result.time, result.materialTemperature().T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, label="Temperature (keV)", ax=ax)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("t (ns)")
    ax.set_title(f"Material temperature, {result.label}")
    _styleAxis(ax)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plotTemperatureMMSLoaded(results, position=None, binIndex=20):
    """Material and radiation temperature vs. time in one cell, against the exact MMS.

    position (cm) defaults to the centre of cell binIndex of the first run; each run uses its own
    cell nearest that x."""
    if not results:
        return None
    position = _defaultPosition(results, binIndex) if position is None else position
    fig, axis = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, result in enumerate(results):
        cell = result.cellAt(position)
        time = result.time
        exactMaterial, exactRadiation = _exactMMS(result.params, result.x[cell], time)
        color = colors[i % len(colors)]
        markEvery = max(1, len(time) // 30)
        axis.plot(time, result.materialTemperature()[cell], color=color, label=f"{result.label} material")
        axis.plot(time, result.radiationTemperature()[cell], color=color, linestyle="--",
                  label=f"{result.label} radiation")
        axis.plot(time, exactMaterial, color=color, linestyle=":", marker="o", markersize=4,
                  markevery=markEvery, label=f"{result.label} MMS material")
        axis.plot(time, exactRadiation, color=color, linestyle=":", marker="s", markersize=4,
                  markevery=markEvery, label=f"{result.label} MMS radiation")

    axis.set_xlabel("Time (ns)")
    axis.set_ylim(0.25, 0.85)
    axis.set_ylabel("Temperature (keV)")
    axis.set_title(f"Temperature at x = {position:.3g} cm: solution vs. MMS")
    _styleAxis(axis)
    axis.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    plt.show()
    return fig, axis


def plotMaterialTemperatureMMSErrorLoaded(results, position=None, binIndex=20):
    """Solved-minus-MMS material temperature vs. time in one cell (same position rule as above)."""
    if not results:
        return None
    position = _defaultPosition(results, binIndex) if position is None else position
    fig, axis = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, result in enumerate(results):
        cell = result.cellAt(position)
        exactMaterial, _ = _exactMMS(result.params, result.x[cell], result.time)
        axis.plot(result.time, result.materialTemperature()[cell] - exactMaterial,
                  color=colors[i % len(colors)],
                  linestyle="--" if result.method == "VMG" else ":",
                  label=result.label)

    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Material temperature error (keV)")
    axis.set_title(f"Material temperature error from MMS at x = {position:.3g} cm")
    _styleAxis(axis)
    axis.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    plt.show()
    return fig, axis


# --- Temperature snapshots (radiation | material profiles at a few times) -------------------
# Color = snapshot time (light -> dark blue), linestyle = run, gray dotted = exact MMS.
SNAPSHOT_RUN_STYLES = ["-", "--", "-.", (0, (5, 1, 1, 1, 1, 1))]   # ":" is reserved for the MMS
SNAPSHOT_REFERENCE = dict(color="0.2", linestyle=":", linewidth=1.3)


def snapshotColors(count):
    """Sequential single-hue ramp (light -> dark blue) for ordered snapshot times."""
    if count <= 1:
        return [plt.cm.Blues(0.85)]
    return [plt.cm.Blues(v) for v in np.linspace(0.4, 0.95, count)]


def progressIndices(temperatureSet, count=5):
    """Time indices at equal fractions of the run's total change (0, 1/(n-1), ..., 1).

    Change = mean over cells of |T(t) - T(0)| (running max), so snapshots land where the solution
    is evolving instead of bunching in a quiet early period of a log time grid.
    """
    T = np.asarray(temperatureSet, dtype=float)
    change = np.maximum.accumulate(np.mean(np.abs(T - T[:, :1]), axis=0))
    nt = T.shape[1]
    if change[-1] <= 0:
        return np.unique(np.linspace(0, nt - 1, count, dtype=int))
    targets = np.linspace(0, 1, count) * change[-1]
    return np.unique([int(np.argmax(change >= target)) for target in targets])


def plotMMSComparison(
    grid, params, axes=None, labelPrefix=None, show=True, includeMMS=True,
    snapshotTimes=None, runIndex=None,
):
    """Radiation (left) and material (right) temperature profiles at a few times.

    Called once per run on shared axes by the *Loaded wrappers, which then add one legend.
    """
    createFigure = axes is None
    if createFigure:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    fig = axes[0].figure
    radiationAxis, materialAxis = axes
    meta = fig.__dict__.setdefault("_snapshotMeta", {"runs": [], "times": None, "mms": False})

    if snapshotTimes is None:
        snapshotTimes = grid.timeSet[progressIndices(grid.temperatureSet, min(5, len(grid.timeSet)))]
    snapshotTimes = np.asarray(snapshotTimes)
    if meta["times"] is None:
        meta["times"] = snapshotTimes
    colors = snapshotColors(len(snapshotTimes))
    if runIndex is None:
        runIndex = len(meta["runs"])

    x = grid.spaceMid
    method = "VMG" if params.movingCoordinates else "SM"
    groups = getattr(params, "freqNum", None)
    runName = labelPrefix or (f"{method} ({groups} groups)" if groups is not None else method)
    style = dict(linestyle=SNAPSHOT_RUN_STYLES[runIndex % len(SNAPSHOT_RUN_STYLES)])
    meta["runs"].append((runName, style))

    for k, target in enumerate(snapshotTimes):
        timeIndex = int(np.argmin(np.abs(grid.timeSet - target)))
        time = grid.timeSet[timeIndex]
        radiationEnergy = np.sum(grid.fullTensorPhiTime[timeIndex], axis=0) / (grid.constants.a * grid.constants.c)
        radiationAxis.plot(x, np.maximum(radiationEnergy, 0) ** 0.25, color=colors[k], **style)
        materialAxis.plot(x, grid.temperatureSet[:, timeIndex], color=colors[k], **style)
        if includeMMS and runIndex == 0:
            spatial = np.exp(-(x**2) / (2 * params.mmsSigmaX**2)) / (params.mmsSigmaX * np.sqrt(2 * np.pi))
            decay = np.exp(-params.mmsTau * time / 2)
            offset = getattr(params, "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET)
            radiationAxis.plot(x, MMS.temperatureFromFourthPowerProfile(params.mmsT0, (1 + decay) * spatial, offset),
                               **SNAPSHOT_REFERENCE, zorder=3)
            materialAxis.plot(x, MMS.temperatureFromFourthPowerProfile(params.mmsT0, (1 - decay) * spatial, offset),
                              **SNAPSHOT_REFERENCE, zorder=3)
            meta["mms"] = True

    radiationAxis.set_title("Radiation temperature")
    materialAxis.set_title("Material temperature")
    for axis in (radiationAxis, materialAxis):
        axis.set_xlabel("Position x (cm)")
        axis.set_ylabel("Temperature (keV)")
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.grid(True, which="major", color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
    if createFigure:
        _addMMSLegend(fig, axes)
        if show:
            plt.show()
    return fig, (radiationAxis, materialAxis)


def _addMMSLegend(fig, axes, fontsize="small"):
    """Legend below the panels: one row of snapshot times, one row of runs (+ exact MMS)."""
    meta = getattr(fig, "_snapshotMeta", None)
    if not meta or meta["times"] is None:
        return
    proxy = lambda label, color="0.35", **kw: Line2D([], [], color=color, label=label, **kw)
    timeRow = [proxy(f"t = {t:.3g} ns", c, linewidth=2.5)
               for t, c in zip(meta["times"], snapshotColors(len(meta["times"])))]
    styleRow = [proxy(name, **style) for name, style in meta["runs"]]
    if meta["mms"]:
        styleRow.append(proxy("exact MMS", **SNAPSHOT_REFERENCE))
    rows = [row for row in (timeRow, styleRow) if row]
    height = 0.06
    fig.tight_layout(rect=(0, height * len(rows) + 0.02, 1, 1))
    for r, row in enumerate(rows):
        legend = fig.legend(handles=row, loc="lower center", ncol=len(row), frameon=False,
                            fontsize=fontsize, bbox_to_anchor=(0.5, height * (len(rows) - 1 - r)))
        fig.add_artist(legend)


def _loadedGrid(result):
    from types import SimpleNamespace
    n = result.timeCount
    if n == 0:
        raise ValueError(f"{result} contains no time samples.")
    return SimpleNamespace(
        spaceMid=result.x,
        fullTensorPhiTime=result.data["fullTensorPhi"][:n],
        temperatureSet=result.materialTemperature(),
        timeSet=result.time,
        freqGrid=np.asarray(result.params["freqGrid"]),
        constants=SimpleNamespace(a=const.a, c=const.c),
    )


def plotMMSComparisonLoaded(
    result, axes=None, show=True, includeMMS=True, snapshotTimes=None, runIndex=None,
):
    """Plot temperature snapshots from one saved SM/VMG result."""
    from types import SimpleNamespace
    params = result.params
    plotParams = SimpleNamespace(movingCoordinates=params["movingCoordinates"], freqNum=params["groups"])
    if includeMMS:
        plotParams.mmsT0 = params["mmsT0"]
        plotParams.mmsSigmaX = params["mmsSigmaX"]
        plotParams.mmsTau = params["mmsTau"]
        plotParams.mmsTemperatureOffset = params["mmsTemperatureOffset"]
    return plotMMSComparison(
        _loadedGrid(result), plotParams, axes=axes, labelPrefix=result.label, show=show,
        includeMMS=includeMMS, snapshotTimes=snapshotTimes, runIndex=runIndex,
    )


def _snapshotsLoaded(results, includeMMS):
    """Common snapshot times (spread over the first run's evolution) so a color is the same time
    for every run."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    first = _loadedGrid(results[0])
    targets = first.timeSet[progressIndices(first.temperatureSet, 5)]
    for i, result in enumerate(results):
        plotMMSComparisonLoaded(result, axes=axes, show=False, includeMMS=includeMMS,
                                snapshotTimes=targets, runIndex=i)
    _addMMSLegend(fig, axes)
    plt.show()
    return fig, axes


def plotMMSComparisonsLoaded(results):
    """Overplot loaded MMS runs at common snapshot times against the exact MMS."""
    if not results:
        return None
    return _snapshotsLoaded(results, includeMMS=True)


def plotTemperatureSnapshotsLoaded(results):
    """Overplot loaded non-MMS temperature profiles at common snapshot times."""
    if not results:
        return None
    return _snapshotsLoaded(results, includeMMS=False)


def plotSpaceTime(phi, params, dt):
    plt.figure(figsize=(8, 5))

    plt.imshow(
        phi,
        aspect='auto',
        origin='lower',
        extent=[params.xMin, params.xMax, 0, params.timeMax]
    )

    plt.colorbar(label='Scalar Flux')
    plt.xlabel('Position x')
    plt.ylabel('Time')
    plt.title('Space-Time Scalar Flux')

    plt.show()

def plotSnapshot(sol, grid, title=None, label="Flux"):
    sol = np.asarray(sol)

    x = grid.spaceMid
    plt.figure()
    plt.plot(x, sol)

    plt.xlabel("Position x")
    plt.ylabel(label)

    if title:
        plt.title(title)

    plt.grid(True)
    plt.show()

def plotSelectSpectra(fullphi, grid, params, cell_idx=0, ax=None, save_path=None):

    freq_centers = grid.freqGroups

    # 2. Extract spectrum array for selected cell
    phi_arr = np.asarray(fullphi)
    if phi_arr.shape[0] == params.freqNum:
        spectrum = phi_arr[:, cell_idx]
    else:
        spectrum = phi_arr[cell_idx, :]

    # 3. Setup plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        standalone = True
    else:
        standalone = False

    ax.plot(freq_centers, spectrum, lw=.4, label=f"Cell {cell_idx} Spectrum")

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'Frequency [1/ns]', fontsize=11)
    ax.set_ylabel(r'Scalar Flux $\phi_g$', fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc='best')

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Spectrum saved to {save_path}")
        plt.show()

    return ax

def plot_spectra_at_times(results, time_indices, position=None, maxFreq=None, binIndex=20):
    """Group scalar flux vs. physical frequency in one cell at a few time indices (VMG groups are
    mapped to nu = u * T). position (cm) defaults to the centre of cell binIndex of the first run."""
    if not results:
        return None
    position = _defaultPosition(results, binIndex) if position is None else position
    fig, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        cell = result.cellAt(position)
        frequencies = np.asarray(result.params["freqGrid"])
        centres = 0.5 * (frequencies[:-1] + frequencies[1:])
        phi = result.data["fullTensorPhi"]
        style = "--" if result.method == "VMG" else "-"
        for t_idx in time_indices:
            if not 0 <= t_idx < result.timeCount:
                print(f"Warning: time index {t_idx} is out of range for {result} and was skipped.")
                continue
            freq = centres * result.materialTemperature()[cell, t_idx] if result.method == "VMG" else centres
            axis.plot(freq, phi[t_idx, :, cell], linestyle=style,
                      label=f"{result.label}, t = {result.time[t_idx]:.3g} ns")

    maxFreq = maxFreq if maxFreq is not None else results[0].params.get("maxFreq", 45)
    axis.set_xlim(0, maxFreq / 10)
    axis.set_xlabel("Photon energy (keV)")
    axis.set_ylabel(r"Group scalar flux $\phi_g$")
    axis.set_title(f"Spectra at x = {position:.3g} cm")
    _styleAxis(axis)
    axis.legend(frameon=False, loc="best", fontsize="small")
    fig.tight_layout()
    plt.show()
    return fig, axis


def analyzeRank(results, time_indices=None, energy_threshold=None, tol=None, plot=True):
    """Rank over time of the (groups x nBins) scalar-flux matrix of each run, from its SVD.

    energy_threshold : list of fractions of the energy sum(s^2) to keep (e.g. [0.99]) on the
                       time-centred data, or
    tol              : list of absolute singular-value cutoffs, or
    neither          : numpy's matrix_rank.
    Returns {result.label: (rankSet, singularValues)}.
    """
    out = {}
    for result in results:
        phi = result.data["fullTensorPhi"][: result.timeCount]
        indices = np.arange(phi.shape[0] - 1) if time_indices is None else np.asarray(time_indices)
        sub = phi[indices]
        times = result.time[indices]

        if energy_threshold is not None:
            singular = np.linalg.svd(sub - np.mean(sub, axis=0), compute_uv=False)
            energy = np.cumsum(singular**2, axis=1)
            total = energy[:, -1:]
            fraction = np.divide(energy, total, out=np.ones_like(energy), where=total > 0)
            rankSet = [np.array([np.searchsorted(f, level) + 1 for f in fraction]) for level in energy_threshold]
            names = [f"{100 * level:g}% energy" for level in energy_threshold]
        elif tol is not None:
            singular = np.linalg.svd(sub, compute_uv=False)
            rankSet = [np.sum(singular > cutoff, axis=1) for cutoff in tol]
            names = [f"cutoff {cutoff:g}" for cutoff in tol]
        else:
            singular = np.linalg.svd(sub, compute_uv=False)
            rankSet = [np.array([np.linalg.matrix_rank(m) for m in sub])]
            names = ["numpy matrix_rank"]
        out[result.label] = (rankSet, singular)

        if plot:
            fig, axis = plt.subplots(figsize=(12, 4.5))
            for ranks, name in zip(rankSet, names):
                axis.plot(times[:-1], ranks[:-1], label=name)
            axis.set_xlabel("t (ns)")
            axis.set_ylabel("Rank")
            axis.set_title(f"Rank of the scalar flux, {result.label}")
            _styleAxis(axis)
            axis.legend(frameon=False, loc="best")
            fig.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(figurePath("rank", f"{result.params['runLabel']}_{timestamp}.pdf"))
            plt.show()
    return out


def analyze_rank_psi(psi_tensor, time_set, time_indices=None, energy_threshold=None, tol=None, plot=True):

    if time_indices is None:
        time_indices = np.arange(psi_tensor.shape[0])
    else:
        time_indices = np.array(time_indices)

    sub_tensor = psi_tensor[time_indices, :, :, :]
    s_matrix = np.linalg.svd(sub_tensor, compute_uv=False)  # shape: (n_selected_times, min(freqNum, nBins))

    ranks = []
    for idx, s in enumerate(s_matrix):
        if energy_threshold is not None:
            # Effective rank based on retained energy cutoff
            cum_energy = np.cumsum(s**2) / np.sum(s**2)
            r = np.searchsorted(cum_energy, energy_threshold) + 1
        elif tol is not None:
            # Rank based on singular value magnitude cutoff
            r = np.sum(s > tol)
        else:
            # Default numerical rank using matrix_rank
            r = np.linalg.matrix_rank(sub_tensor[idx])
        ranks.append(r)

    ranks = np.array(ranks)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Rank vs. Time
        if time_indices is None:
            time_indices = np.arange(psi_tensor.shape[0])
        ax1.plot(np.take(time_set, time_indices)[:-1], ranks[:-1], color="tab:blue")
        ax1.set_xlabel("Time Step Index")
        ax1.set_ylabel("Rank")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Singular Value Decay
        for idx, t_idx in enumerate(time_indices):
            ax2.semilogy(s_matrix[idx], label=f"t = {t_idx}")
        ax2.set_xlabel("Singular Value Index")
        ax2.set_ylabel("Singular Value Magnitude (log scale)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        if len(time_indices) <= 10:
            ax2.legend()

        plt.tight_layout()
        plt.show()

    return ranks, s_matrix


# --- Manufactured solution (MMS) demonstration plots -------------------------------------------
def MMSPlot():
    tMin, tMax = 0, 1.0
    tNum = 1000
    T0 = 0.25
    tSet = np.linspace(tMin, tMax, tNum)
    tSet, Trad, Tmat = MMS.MMSBase(tSet, T0, 0)
    plt.plot(tSet, Tmat, label="MMS Material")
    plt.plot(tSet, Trad, label="MMS Radiation Temperature")
    plt.xlabel("t (ns)")
    plt.ylabel("Temperature (keV)")
    plt.minorticks_on()

    plt.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True
    )

    plt.legend(
        frameon=False,
        loc="best"
    )
    plt.show()

def MMSPlot1D(
    T0=0.25,
    sigmaX=0.5,
    t=0.2,
    tau=2,
    xMin=-2.0,
    xMax=2.0,
    xNum=500,
    savePath=None,
    temperatureOffset=MMS.DEFAULT_TEMPERATURE_OFFSET,
):
    """Plot the Gaussian 1-D manufactured temperatures and spatial profile."""
    x = np.linspace(xMin, xMax, xNum)
    expTerm = np.exp(-tau * t / 2)
    spatial = np.exp(-(x**2) / (2 * sigmaX**2)) / (
        sigmaX * np.sqrt(2 * np.pi)
    )
    Trad = MMS.temperatureFromFourthPowerProfile(
        T0, (1 + expTerm) * spatial, temperatureOffset
    )
    Tmat = MMS.temperatureFromFourthPowerProfile(
        T0, (1 - expTerm) * spatial, temperatureOffset
    )

    fig, (temperatureAxis, spatialAxis) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7), constrained_layout=True
    )
    temperatureAxis.plot(x, Trad, label="Radiation temperature")
    temperatureAxis.plot(x, Tmat, label="Material temperature")
    temperatureAxis.set_ylabel("Temperature (keV)")
    temperatureAxis.set_title(f"1-D Gaussian MMS solution at t = {t:g} ns")
    temperatureAxis.legend(frameon=False, loc="best")
    temperatureAxis.minorticks_on()
    temperatureAxis.tick_params(
        which="both", direction="in", top=True, right=True
    )

    spatialAxis.plot(x, spatial, color="black", label="Gaussian f(x)")
    spatialAxis.set_xlabel("x (cm)")
    spatialAxis.set_ylabel("f(x)")
    spatialAxis.legend(frameon=False, loc="best")
    spatialAxis.minorticks_on()
    spatialAxis.tick_params(
        which="both", direction="in", top=True, right=True
    )
    plt.show()


def MMSPlot1DTimeEvolution(
    T0=0.25,
    sigmaX=0.5,
    tau=2,
    tMin=0.0,
    tMax=1.0,
    tNum=300,
    xMin=-2.0,
    xMax=2.0,
    xNum=300,
    savePath=None,
    temperatureOffset=MMS.DEFAULT_TEMPERATURE_OFFSET,
):
    """Plot radiation and material temperature over space and time."""
    x = np.linspace(xMin, xMax, xNum)
    t = np.linspace(tMin, tMax, tNum)
    spatial = np.exp(-(x**2) / (2 * sigmaX**2)) / (
        sigmaX * np.sqrt(2 * np.pi)
    )
    expTerm = np.exp(-tau * t / 2)
    Trad = MMS.temperatureFromFourthPowerProfile(
        T0,
        (1 + expTerm[:, None]) * spatial[None, :],
        temperatureOffset,
    )
    Tmat = MMS.temperatureFromFourthPowerProfile(
        T0,
        (1 - expTerm[:, None]) * spatial[None, :],
        temperatureOffset,
    )

    fig, (radiationAxis, materialAxis) = plt.subplots(
        2, 1, sharex=True, sharey=True, figsize=(8, 7),
        constrained_layout=True,
    )
    radiationPlot = radiationAxis.pcolormesh(
        x, t, Trad, shading="auto", cmap="inferno"
    )
    materialPlot = materialAxis.pcolormesh(
        x, t, Tmat, shading="auto", cmap="inferno"
    )

    radiationAxis.set_title("Radiation temperature")
    materialAxis.set_title("Material temperature")
    materialAxis.set_xlabel("x (cm)")
    radiationAxis.set_ylabel("t (ns)")
    materialAxis.set_ylabel("t (ns)")
    fig.colorbar(radiationPlot, ax=radiationAxis, label="Temperature (keV)")
    fig.colorbar(materialPlot, ax=materialAxis, label="Temperature (keV)")
    fig.suptitle("1-D Gaussian MMS temperature evolution")

    for axis in (radiationAxis, materialAxis):
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
    plt.show()


if __name__ == "__main__":
    MMSPlot1D()
    MMSPlot1DTimeEvolution()
