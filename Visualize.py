import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from matplotlib.animation import FuncAnimation
from Base import Constants as const
import DiffEqSolve as diffEq
import MMS

# Plot Stuff
import os
import tkinter as tk
from tkinter import filedialog
from cycler import cycler


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
    filename = f"figures/{label}_{timestamp}.pdf"
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
    filename = f"figures/Temperature_vs_Time_{timestamp}.pdf"
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


    solution = diffEq.IMSolve(inputDictTest)
    t = solution.t
    T = solution.y[-1]
    Tr = (np.sum(solution.y[:-1],axis=0)/ const.a / const.c)**.25
    return t, T, Tr

def plotTemperatureLoaded(dataSet, paramsSet, fileSet, folderSet, const):
    for i, data in enumerate(dataSet):
        params = paramsSet[i]
        file = fileSet[i]
        folder = folderSet[i]
        if folder == "InfiniteMedium":
            methodName = "MG"
        else:
            methodName = "VMG"

        timeCount = min(
            len(data["timeSet"]),
            len(data["fullTensorPhi"]),
            data["temperatureSet"].shape[1],
        )
        t = data["timeSet"][:timeCount]
        timeSteps = timeCount - 1
        cellIndex = params["nBins"] // 2
        radiationFlux = np.sum(
            data["fullTensorPhi"][:timeCount, :, cellIndex], axis=1
        )
        Trad = (radiationFlux / const.a / const.c) ** 0.25
        labelT = f'{file} using {methodName} for {params["groups"]} groups with {timeSteps} steps'
        T = data["temperatureSet"][cellIndex, :timeCount]
        plt.plot(t, T, label=f"{labelT} material")
        plt.plot(
            t, Trad, linestyle='--',
            label=f"{labelT} radiation",
        )

    reflectiveResults = [
        (data, params)
        for data, params in zip(dataSet, paramsSet)
        if params.get("boundaryLeft") == "Reflective"
        and params.get("boundaryRight") == "Reflective"
    ]
    if reflectiveResults:
        maxTime = max(np.max(data["timeSet"]) for data, _ in reflectiveResults)
        t, T, Tr = getDiffeq(const, maxTime)
        plt.plot(t, T, label="0-D Solve (reflective boundaries)", linestyle="--")
        plt.plot(t, Tr, label="0-D Rad Temperature (reflective boundaries)")

    plt.ylim(0, 0.8)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/TempPlot_{timestamp}.pdf"
    plt.savefig(filename)
    plt.show()

    # plt.plot(np.linspace(0, 1, timeSteps-1), dt)
    # plt.show
  

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

def plotTemperatureTimeLoaded(dataSet, paramsSet, fileSet, folderSet):
    data, params = dataSet[0], paramsSet[0]
    x = params["spaceMid"]  # cell centers
    time = data["timeSet"]
    T_time = data["temperatureSet"]  # shape: (nBins, nSteps+1)

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
    plt.tight_layout()
    plt.show()


def plotTemperatureMMSLoaded(results, binIndex=20):
    """Compare loaded MMS material and radiation temperatures at one spatial bin."""
    if not results:
        return None

    fig, axis = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for runIndex, (data, params, label) in enumerate(results):
        x = np.asarray(params["spaceMid"])
        if not 0 <= binIndex < len(x):
            raise IndexError(
                f"Bin index {binIndex} is outside the saved spatial grid "
                f"of {len(x)} bins."
            )

        timeCount = min(
            len(data["timeSet"]),
            data["fullTensorPhi"].shape[0],
            data["temperatureSet"].shape[1],
        )
        if timeCount == 0:
            raise ValueError(f"Saved MMS result {label!r} contains no time samples.")

        time = np.asarray(data["timeSet"])[:timeCount]
        materialTemperature = data["temperatureSet"][binIndex, :timeCount]
        radiationEnergy = (
            np.sum(data["fullTensorPhi"][:timeCount, :, binIndex], axis=1)
            / (const.a * const.c)
        )
        radiationTemperature = np.maximum(radiationEnergy, 0) ** 0.25

        t0 = params["mmsT0"]
        sigmaX = params["mmsSigmaX"]
        tau = params["mmsTau"]
        temperatureOffset = params.get(
            "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET
        )
        spatial = np.exp(-(x[binIndex] ** 2) / (2 * sigmaX**2)) / (
            sigmaX * np.sqrt(2 * np.pi)
        )
        decay = np.exp(-tau * time / 2)
        exactMaterial = MMS.temperatureFromFourthPowerProfile(
            t0, (1 - decay) * spatial, temperatureOffset
        )
        exactRadiation = MMS.temperatureFromFourthPowerProfile(
            t0, (1 + decay) * spatial, temperatureOffset
        )
        color = colors[runIndex % len(colors)]
        markEvery = max(1, timeCount // 30)

        axis.plot(
            time, materialTemperature, color=color,
            label=f"{label} material",
        )
        axis.plot(
            time, radiationTemperature, color=color, linestyle="--",
            label=f"{label} radiation",
        )
        axis.plot(
            time, exactMaterial, color=color, linestyle=":",
            marker="o", markersize=4, markevery=markEvery,
            label=f"{label} MMS material",
        )
        axis.plot(
            time, exactRadiation, color=color, linestyle=":",
            marker="s", markersize=4, markevery=markEvery,
            label=f"{label} MMS radiation",
        )

    axis.set_xlabel("Time (ns)")
    axis.set_ylim(0.25, 0.85)
    axis.set_ylabel("Temperature (keV)")
    axis.set_title(f"Temperature at spatial bin {binIndex}: solution vs. MMS")
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    plt.show()
    return fig, axis


def plotMaterialTemperatureMMSErrorLoaded(results, binIndex=20):
    """Plot solved-minus-MMS material-temperature error at one spatial bin."""
    if not results:
        return None

    fig, axis = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for runIndex, (data, params, label) in enumerate(results):
        x = np.asarray(params["spaceMid"])
        if not 0 <= binIndex < len(x):
            raise IndexError(
                f"Bin index {binIndex} is outside the saved spatial grid "
                f"of {len(x)} bins."
            )

        timeCount = min(
            len(data["timeSet"]),
            data["temperatureSet"].shape[1],
        )
        if timeCount == 0:
            raise ValueError(f"Saved MMS result {label!r} contains no time samples.")

        time = np.asarray(data["timeSet"])[:timeCount]
        materialTemperature = data["temperatureSet"][binIndex, :timeCount]
        t0 = params["mmsT0"]
        sigmaX = params["mmsSigmaX"]
        tau = params["mmsTau"]
        temperatureOffset = params.get(
            "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET
        )
        spatial = np.exp(-(x[binIndex] ** 2) / (2 * sigmaX**2)) / (
            sigmaX * np.sqrt(2 * np.pi)
        )
        exactMaterial = MMS.temperatureFromFourthPowerProfile(
            t0,
            (1 - np.exp(-tau * time / 2)) * spatial,
            temperatureOffset,
        )

        axis.plot(
            time,
            materialTemperature - exactMaterial,
            color=colors[runIndex % len(colors)],
            linestyle="--" if params.get("movingCoordinates", False) else ":",
            label=label,
        )

    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Material temperature error (keV)")
    axis.set_title(
        f"Material temperature error from MMS at spatial bin {binIndex}"
    )
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)
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
    methodName = "VMG" if params.movingCoordinates else "MG"
    groups = getattr(params, "freqNum", None)
    runName = labelPrefix or (f"{methodName} ({groups} groups)" if groups is not None else methodName)
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


def _loadedGrid(data, params):
    from types import SimpleNamespace
    timeCount = min(len(data["timeSet"]), data["fullTensorPhi"].shape[0], data["temperatureSet"].shape[1])
    if timeCount == 0:
        raise ValueError("A saved result contains no time samples.")
    return SimpleNamespace(
        spaceMid=np.asarray(params["spaceMid"]),
        fullTensorPhiTime=data["fullTensorPhi"][:timeCount],
        temperatureSet=data["temperatureSet"][:, :timeCount],
        timeSet=np.asarray(data["timeSet"])[:timeCount],
        freqGrid=np.asarray(params["freqGrid"]),
        constants=SimpleNamespace(a=const.a, c=const.c),
    )


def plotMMSComparisonLoaded(
    data, params, axes=None, labelPrefix=None, show=True, includeMMS=True,
    snapshotTimes=None, runIndex=None,
):
    """Plot temperature snapshots from one saved MG/VMG result."""
    from types import SimpleNamespace
    plotParams = SimpleNamespace(movingCoordinates=params["movingCoordinates"], freqNum=params.get("groups"))
    if includeMMS:
        plotParams.mmsT0 = params["mmsT0"]
        plotParams.mmsSigmaX = params["mmsSigmaX"]
        plotParams.mmsTau = params["mmsTau"]
        plotParams.mmsTemperatureOffset = params.get("mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET)
    return plotMMSComparison(
        _loadedGrid(data, params), plotParams, axes=axes, labelPrefix=labelPrefix, show=show,
        includeMMS=includeMMS, snapshotTimes=snapshotTimes, runIndex=runIndex,
    )


def _snapshotsLoaded(results, includeMMS):
    """Common snapshot times (spread over the first run's evolution) so a color is the same time
    for every run."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    first = _loadedGrid(results[0][0], results[0][1])
    targets = first.timeSet[progressIndices(first.temperatureSet, 5)]
    for i, (data, params, label) in enumerate(results):
        plotMMSComparisonLoaded(data, params, axes=axes, labelPrefix=label, show=False,
                                includeMMS=includeMMS, snapshotTimes=targets, runIndex=i)
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


def plotFinalFlux(grid, label="Final Scalar Flux"):
    x = grid.spaceMid
    phi_by_group = grid.fullTensorPhiTime[-1]  # shape: (nFreq, nBins)

    plt.figure()
    for group, phi in enumerate(phi_by_group):
        plt.plot(x, phi, label=f"Group {group + 1}")

    # if phi_by_group.shape[0] > 1:
    #     plt.plot(x, np.sum(phi_by_group, axis=0), "k--", label="Total")

    plt.xlabel("Position x")
    plt.ylabel("Scalar Flux")
    plt.title(label)
    plt.legend()
    plt.show()


def plotOutScalar(phil, grid, constants, label="Scalar Flux φ"):

    x = 0.5 * (grid.grid[:-1] + grid.grid[1:])
    scalar = np.sum(grid.w[:, None] * phil, axis=0)
    split_idx = np.searchsorted(x, 0)

    x = x[split_idx:]
    sol_half = scalar[split_idx:] # Slice the spatial dimension (axis 1)

    plt.plot(x, sol_half, label=label)
    plt.xlabel("Position x")
    plt.ylabel("Scalar Flux φ")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/{label}_{timestamp}.pdf"
    plt.savefig(filename)
    plt.show()

    #Plot T_r for comparison
    T_r = (sol_half / constants.a / constants.c)**0.25
    plt.plot(x, T_r, label=f"{label} T_r")  
    plt.xlabel("Position x")
    plt.ylabel("Radiation Temperature T_r")
    plt.legend()
    plt.show()

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

def animate_solution(fullPhi, params, grid, interval=15):
    sol = np.array(fullPhi)   # shape: (Nt, Nx)
    x_full = np.linspace(params.xMin, params.xMax, sol.shape[1])

    # 1. Find the index where x >= 0
    # np.searchsorted finds the first index where x would be inserted to maintain order
    split_idx = np.searchsorted(x_full, 0)

    # 2. Slice the arrays to only include x >= 0
    x = x_full[split_idx:]
    sol_half = sol[:, split_idx:] # Slice the spatial dimension (axis 1)

    fig, ax = plt.subplots()
    line, = ax.plot(x, sol_half[0])

    # 3. Update plot limits for the new range
    ax.set_xlim(0, params.xMax)
    ax.set_ylim(np.min(sol_half), np.max(sol_half))
    ax.set_xlabel('x')
    ax.set_ylabel('Scalar Flux')

    def update(frame):
        line.set_ydata(sol_half[frame])
        # Using frame * dt if grid.timeSet isn't exactly matched to Nt
        ax.set_title(f'Time: {frame * grid.dt[frame]:.2f}')
        return line,

    anim = FuncAnimation(
        fig,
        update,
        frames=sol_half.shape[0],
        interval=interval
    )
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

def plot_spectra_at_times(dataSet, time_indices, paramsSet, fileSet, folderSet, bin_idx, maxFreq, freqs=None):
    plt.figure(figsize=(8, 5))
    for i, data in enumerate(dataSet):
        params = paramsSet[i]
        file = fileSet[i]
        folder = folderSet[i]
        temp = data["temperatureSet"]
        frequencies = params["freqGrid"]
        freqBase = 0.5 * (frequencies[:-1] + frequencies[1:])  # Midpoint of frequency groups for plotting
        if folder == "InfiniteMedium":
            methodName = "SM"
            style = "-"
        else:
            style = "--"
            methodName = "VCM"
        phi_tensor = data["fullTensorPhi"]
        n_steps, freq_num, n_bins = phi_tensor.shape
        if not (0 <= bin_idx < n_bins):
            raise ValueError(f"bin_idx {bin_idx} out of bounds for array with {n_bins} bins.")

        timeSteps = np.shape(data["timeSet"])[0]

        for t_idx in time_indices:
            if 0 <= t_idx < n_steps:
                if methodName == "VCM":
                    freq = freqBase * temp[bin_idx, t_idx]  # Scale frequency by temperature for VCM method
                    spectrum = phi_tensor[t_idx, :, bin_idx]
                else:
                    spectrum = phi_tensor[t_idx, :, bin_idx]
                    print(f"Time {t_idx} - Min: {np.nanmin(spectrum)}, Max: {np.nanmax(spectrum)}, NaNs: {np.isnan(spectrum).sum()}")
                    freq = freqBase
                plt.plot(freq, spectrum, linestyle=style, label=f'Time: {t_idx}, {methodName} with {file} init {params["groups"]} groups for {timeSteps} steps')
            else:
                print(f"Warning: Time step index {t_idx} is out of bounds (max {n_steps - 1}) and skipped.")

    plt.xlabel("Frequency" if freqs is not None else "Frequency Index")
    plt.ylabel("Spectrum Magnitude ($\Phi$)")
    plt.xlim(0, maxFreq/10)  # Adjust x-axis limit based on your frequency range
    plt.title(f"Spectra Comparison for Bin {bin_idx}")
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
    plt.tight_layout()
    plt.show()

def analyzeRank(dataSet, time_set, paramsSet, fileSet, folderSet, time_indices=None, energy_threshold=None, tol=None, plot=True):
    """
    Performs SVD on slices of shape (freqNum, nBins) at specified time steps to compute rank.

    Parameters:
    -----------
    phi_tensor : np.ndarray
        3D solution tensor of shape (nSteps + 1, freqNum, nBins).
    time_indices : list or array-like, optional
        Time step indices to analyze. If None, analyzes all time steps.
    energy_threshold : float or None, optional
        Fraction of cumulative variance/energy (sum(s^2)) to define effective rank (e.g., 0.99 for 99%).
        Set to None to use numerical tolerance instead.
    tol : float or None, optional
        Absolute singular value threshold (s > tol). If both energy_threshold and tol are None,
        uses numpy's default matrix_rank tolerance.
    plot : bool
        If True, plots effective rank over time and the singular value spectra.

    Returns:
    --------
    ranks : np.ndarray
        Computed ranks for each evaluated time step.
    s_matrix : np.ndarray
        Singular values for each analyzed time step, shape (len(time_indices), min(freqNum, nBins)).
    """
    for i, data in enumerate(dataSet):
        params = paramsSet[i]
        file = fileSet[i]
        folder = folderSet[i]
        if folder == "InfiniteMedium":
            methodName = "SM"
        else:
            methodName = "VCM"

        phi_tensor = data["fullTensorPhi"]


        if time_indices is None:
            time_indices = np.arange(phi_tensor.shape[0]-1)
        else:
            time_indices = np.array(time_indices)

        # Extract sub-tensor at requested times and compute SVD in batch

        sub_tensor = phi_tensor[time_indices, :, :]
        s_matrix = np.linalg.svd(sub_tensor, compute_uv=False)  # shape: (n_selected_times, min(freqNum, nBins))


        if energy_threshold is None and tol is not None:
            label = 'Tol Threshold'
            rankSet = []
            for t in tol:
                ranks = []
                for idx, s in enumerate(s_matrix):
                    r = np.sum(s > t)
                    ranks.append(r)
                rankSet.append(ranks)



        if energy_threshold is not None and tol is None:
            array = sub_tensor - np.mean(sub_tensor, axis=0)  # Center the data by subtracting the mean across time
            s_matrix = np.linalg.svd(array, compute_uv=False)  # shape: (n_selected_times, min(freqNum, nBins))
            rankSet = []
            label = 'Percentage Threshold'
            for energy in energy_threshold:
                ranks = []
                for idx, s in enumerate(s_matrix):
                    try:
                        with np.errstate(divide='raise', invalid='raise', over='raise'):
                            cum_energy = np.cumsum(s**2) / np.sum(s**2)
                            r = np.searchsorted(cum_energy, energy) + 1
                            ranks.append(r)
                    except FloatingPointError:
                        print("Floating-point error during SVD truncation")
                        print(f' Numerator: {np.cumsum(s**2)}')
                        print(f' Denominator: {np.sum(s**2)}')

                ranks = np.array(ranks)
                rankSet.append(ranks)
        
        if energy_threshold is None and tol is None:
            label = 'Numpy calc'
            rankSet = []
            ranks = []
            if energy_threshold is None and tol is None:
                label = 'Numpy calc'
                ranks = [np.linalg.matrix_rank(s) for s in sub_tensor]
                rankSet = [np.array(ranks)] # Now a 1D array of ranks over time
        


        if plot:

            fig, ax1 = plt.subplots(figsize=(12, 4.5))

            # Rank vs. Time
            if time_indices is None:
                time_indices = np.arange(phi_tensor.shape[0])
            for i, ranks in enumerate(rankSet):

                if energy_threshold is None and tol is None:
                    ax1.plot(np.take(time_set, time_indices)[:-1], ranks[:-1], label=f'NP matrix_rank of {methodName} {params["groups"]} groups')
                elif energy_threshold is None:
                    ax1.plot(np.take(time_set, time_indices)[:-1], ranks[:-1], label=f'Rank at cutoff of {tol[i]} for {methodName} {params["groups"]} groups')
                else:
                    ax1.plot(np.take(time_set, time_indices)[:-1], ranks[:-1], label=f'Rank at cutoff of {100 * energy_threshold[i]} % for {methodName} {params["groups"]} groups' )
        ax1.set_xlabel("Time Step Index")
        ax1.set_ylabel("Rank")


            # # Singular Value Decay
            # for idx, t_idx in enumerate(time_indices):
            #     ax2.semilogy(s_matrix[idx], label=f"t = {t_idx}")
            # ax2.set_xlabel("Singular Value Index")
            # ax2.set_ylabel("Singular Value Magnitude (log scale)")
            # ax2.grid(True, linestyle="--", alpha=0.6)
            # if len(time_indices) <= 5:
            #     ax2.legend()
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
        saveName = params['runLabel']
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figures/{saveName}_{timestamp}.pdf"
        plt.savefig(filename)
        plt.show()

        return rankSet, s_matrix

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


    def readResults(filepath, cls):
        """Reads a text file and restores both the array and the class instance."""
        with open(filepath, "r") as file:
            data = json.load(file)

        restored_array = data["array"]
        restored_instance = cls(**data["object"])

        return restored_array, restored_instance