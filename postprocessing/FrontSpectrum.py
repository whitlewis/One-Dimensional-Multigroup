"""Evolution of the radiation spectrum in one cell as a Marshak wave front passes through it.

    python postprocessing/FrontSpectrum.py                    # pick .h5 files in a dialog
    python postprocessing/FrontSpectrum.py run1.h5 run2.h5    # explicit files (e.g. SM and VMG)
    python postprocessing/FrontSpectrum.py --set PlanckianMarshak --groups 100   # SM | VMG at G=100
options:
    --x 0.3        cell position (cm); default: the cell the front reaches at half the run time
    --n 8          number of sampled spectra
    --groups 100   with --set: only runs with this many groups (one column per run, keep it to a few)
    --planck       overlay the Planckian at each spectrum's material temperature (dotted)
    --emin/--emax  photon-energy range (keV); default 0.01 (or below a Delta/Bimodal spike) to maxFreq

One column per run (e.g. SM | VMG, sharing axes), three panels each:
    top     spectral density phi_nu = phi_g / dnu_g against photon energy, drawn per group, at the
            initial time, the front's arrival (1% of the cell's temperature rise) and evenly spaced
            fractions of the rise up to its peak, coloured by the material temperature then;
    middle  spectrogram: share of the cell's radiation energy per ln(nu) against time and photon
            energy (shape only, the overall level divided out), with the peak of a Planckian at
            the local material temperature (dashed, 3.92 T) and the sampled times (vertical lines);
    bottom  material and radiation temperature in the cell against the same time axis.

VMG groups live in u = nu / T; they are mapped to photon energy with the cell's material
temperature at that time (nu = u * T), so SM and VMG spectra share one axis. Dividing by the group
width makes groups of different widths comparable; the equilibrium value is 4 pi B_nu(T), whose
integral over nu is a c T^4.
"""
import argparse
import os
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("src", "postprocessing"):
    if os.path.join(_ROOT, _folder) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _folder))

import numpy as np

import Base
from Base import Constants as const
import Results
import Visualize as Vis  # applies the project's matplotlib style
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

# One hue, light -> dark with temperature (the lightest end is dropped so every curve is visible)
TEMPERATURE_CMAP = LinearSegmentedColormap.from_list(
    "frontT", plt.get_cmap("Oranges")(np.linspace(0.3, 1.0, 256)))


# --- Physics helpers ---------------------------------------------------------------------------
def groupEdges(result, cell, timeIndex):
    """Photon-energy group edges (keV) in one cell at one time."""
    edges = np.asarray(result.params["freqGrid"], dtype=float)
    if result.method == "VMG":
        edges = edges * result.materialTemperature()[cell, timeIndex]
    return edges


def spectralDensity(result, cell, timeIndex):
    """(edges, phi_g / dnu_g) in one cell at one time."""
    edges = groupEdges(result, cell, timeIndex)
    phi = result.data["fullTensorPhi"][timeIndex, :, cell]
    return edges, phi / np.diff(edges)


def planckDensity(energy, temperature):
    """Equilibrium spectral density 4 pi B_nu(T) (same units as phi_g / dnu_g)."""
    x = np.minimum(const.h * energy / temperature, 700.0)
    return 15.0 * const.a * const.c / np.pi**4 * energy**3 / np.expm1(x)


SPIKE_BOUNDARIES = ("Delta", "Bimodal")


def boundaryName(result):
    """The driving boundary, e.g. "Planckian" (Reflective/Vacuum sides are left out)."""
    sides = [result.params.get(k, "") for k in ("boundaryLeft", "boundaryRight")]
    driving = [s for s in sides if s not in ("Reflective", "Vacuum", "")]
    return "/".join(driving) or "/".join(sides)


def spikeEnergy(result):
    """Photon energy (keV) of the delta spike of a Delta or Bimodal boundary, else None."""
    sides = (result.params.get("boundaryLeft"), result.params.get("boundaryRight"))
    if any(s in SPIKE_BOUNDARIES for s in sides) and result.params.get("deltaFrequency"):
        return float(result.params["deltaFrequency"])
    return None


# --- Choosing the cell and the times -----------------------------------------------------------
def frontPosition(result):
    """x (cm) of the cell the wave front reaches at half the run time.

    The front is where the material temperature first passes halfway between its lowest and
    highest values anywhere in the run."""
    temperature = result.materialTemperature()
    halfway = 0.5 * (temperature.min() + temperature.max())
    reached = temperature >= halfway
    crossed = reached.any(axis=1)
    if not crossed.any():
        return float(result.x[len(result.x) // 2])
    arrival = np.where(crossed, np.argmax(reached, axis=1), -1)
    target = 0.5 * result.time[-1]
    candidates = np.flatnonzero(crossed)
    best = candidates[np.argmin(np.abs(result.time[arrival[candidates]] - target))]
    return float(result.x[best])


def frontTimes(result, cell, count):
    """Time indices where the cell's material temperature first reaches fractions of its rise
    from the starting value to its peak: 0 (initial state), 1% (the front's arrival, when the
    precursor photons are there but the material has barely heated), then evenly spaced to 100%."""
    temperature = result.materialTemperature()[cell]
    rising = np.maximum.accumulate(temperature)
    fractions = np.concatenate(([0.0, 0.01], np.linspace(0.0, 1.0, max(count - 1, 2))[1:]))
    levels = temperature[0] + fractions * (rising[-1] - temperature[0])
    indices = np.minimum(np.searchsorted(rising, levels), result.timeCount - 1)
    if rising[-1] - temperature[0] < 1e-3 * max(abs(temperature[0]), 1e-12):
        print(f"Warning: the temperature at x = {result.x[cell]:.3g} cm barely changes in {result}; "
              "the front may not reach this cell.")
    return np.unique(indices)


# --- Plot --------------------------------------------------------------------------------------
PLANCK_PEAK = 3.9207   # x of the peak of x^4 / (e^x - 1): nu B_nu(T) peaks at nu = 3.92 T
SPECTROGRAM_CMAP = LinearSegmentedColormap.from_list(
    "frontSpec", plt.get_cmap("Blues")(np.linspace(0.08, 1.0, 256)))


def energyShare(result, cell):
    """(edges (G+1, nTime), share (G, nTime)): fraction of the cell's radiation energy per unit
    ln(nu) in each group at every time; independent of group widths and of the overall level."""
    phi = np.asarray(result.data["fullTensorPhi"][: result.timeCount, :, cell], dtype=float).T
    edges = np.asarray(result.params["freqGrid"], dtype=float)[:, None]
    if result.method == "VMG":
        edges = edges * result.materialTemperature()[cell][None, :]
    else:
        edges = np.repeat(edges, phi.shape[1], axis=1)
    total = np.sum(phi, axis=0, keepdims=True)
    share = np.divide(phi, np.log(edges[1:] / edges[:-1]) * total,
                      out=np.zeros_like(phi), where=total > 0)
    return edges, share


def _spectraPanel(axis, result, cell, indices, norm, planck):
    temperature = result.materialTemperature()[cell]
    peak = 0.0
    for index in indices:
        color = TEMPERATURE_CMAP(norm(temperature[index]))
        edges, density = spectralDensity(result, cell, index)
        axis.stairs(density, edges, color=color, linewidth=1.4)
        peak = max(peak, np.nanmax(density))
        if planck:
            energy = np.geomspace(max(edges[0], 1e-6), edges[-1], 400)
            axis.plot(energy, planckDensity(energy, temperature[index]), color=color,
                      linestyle=":", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Photon energy (keV)")
    axis.set_ylabel(r"$\phi_\nu = \phi_g / \Delta\nu_g$")
    axis.set_title(f"{boundaryName(result)} · {result.label}", fontsize="medium")
    spike = spikeEnergy(result)
    if spike:
        axis.axvline(spike, color="gray", linewidth=0.8, label=f"delta spike ({spike:.3g} keV)")
    if planck:
        axis.plot([], [], color="gray", linestyle=":", label=r"$4\pi B_\nu(T)$")
    if planck or spike:
        axis.legend(frameon=False, loc="lower center", fontsize="small")
    Vis._styleAxis(axis)
    return peak


def _spectrogramPanel(axis, result, cell, indices, norm, shareNorm):
    edges, share = energyShare(result, cell)
    time = result.time
    times = np.broadcast_to(time[None, :], edges.shape)
    mesh = axis.pcolormesh(times, edges, share[:, :-1], cmap=SPECTROGRAM_CMAP, norm=shareNorm,
                           shading="flat", rasterized=True)
    temperature = result.materialTemperature()[cell]
    axis.plot(time, PLANCK_PEAK * temperature, color="black", linestyle="--", linewidth=1.2,
              label=r"Planck peak at local $T$ ($3.92\,T$)")
    for index in indices:
        axis.axvline(time[index], color=TEMPERATURE_CMAP(norm(temperature[index])), linewidth=0.9,
                     alpha=0.8)
    axis.set_yscale("log")
    axis.set_ylabel("Photon energy (keV)")
    axis.set_title(f"Spectrum vs. time at x = {result.x[cell]:.3g} cm", fontsize="medium")
    axis.legend(frameon=False, loc="upper left", fontsize="small")
    axis.tick_params(which="both", direction="in", top=True, right=True)
    return mesh


def _historyPanel(axis, result, cell, indices, norm):
    time = result.time
    temperature = result.materialTemperature()[cell]
    axis.plot(time, temperature, color="black", label="Material $T$")
    axis.plot(time, result.radiationTemperature()[cell], color="gray", linestyle="--",
              label="Radiation $T_r$")
    axis.scatter(time[indices], temperature[indices], c=temperature[indices], cmap=TEMPERATURE_CMAP,
                 norm=norm, s=45, zorder=3, edgecolors="black", linewidths=0.5)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Temperature (keV)")
    Vis._styleAxis(axis)
    axis.legend(frameon=False, loc="lower right", fontsize="small")


def plotFrontSpectra(results, position=None, count=8, planck=False, energyRange=None, save=True):
    """One column per run (e.g. SM | VMG), three panels each:
        top     spectra at the times the cell's material temperature reaches evenly spaced
                fractions of its rise, coloured by that temperature;
        middle  spectrogram: share of the radiation energy per ln(nu) against time and photon
                energy, with the Planck peak at the local temperature (dashed) and the sampled
                times (vertical lines);
        bottom  material and radiation temperature in the cell, sharing the time axis.
    Returns the figure."""
    if not results:
        return None
    if len(results) > 4:
        print(f"Warning: {len(results)} runs give {len(results)} columns; pick a few (e.g. --groups).")
    position = frontPosition(results[0]) if position is None else position

    samples = []
    for result in results:
        cell = result.cellAt(position)
        samples.append((result, cell, frontTimes(result, cell, count)))
    sampledT = np.concatenate([r.materialTemperature()[c, idx] for r, c, idx in samples])
    norm = Normalize(vmin=sampledT.min(), vmax=sampledT.max())
    shareMax = max(np.max(energyShare(r, c)[1]) for r, c, _ in samples)
    shareNorm = LogNorm(vmin=shareMax * 1e-4, vmax=shareMax, clip=True)

    low, high = energyRange if energyRange else (None, None)
    spikes = [e for e in map(spikeEnergy, results) if e]
    low = low or min([1e-2] + [0.5 * e for e in spikes])
    high = high or results[0].params.get("maxFreq", 45)

    columns = len(results)
    fig, axes = plt.subplots(3, columns, figsize=(6.2 * columns + 1.2, 11), squeeze=False,
                             gridspec_kw=dict(height_ratios=(1.15, 1.15, 0.7)), layout="constrained")
    peak = 0.0
    for column, (result, cell, indices) in enumerate(samples):
        specAxis, mapAxis, histAxis = axes[:, column]
        peak = max(peak, _spectraPanel(specAxis, result, cell, indices, norm, planck))
        mesh = _spectrogramPanel(mapAxis, result, cell, indices, norm, shareNorm)
        _historyPanel(histAxis, result, cell, indices, norm)
        mapAxis.sharex(histAxis)
        mapAxis.tick_params(labelbottom=False)
        mapAxis.set_ylim(low, high)
        specAxis.set_xlim(low, high)
        if column:   # later columns share the first column's y axes
            for row, axis in enumerate(axes[:, column]):
                axis.sharey(axes[row, 0])
                axis.set_ylabel("")
    for specAxis in axes[0]:
        specAxis.set_ylim(peak * 1e-7, peak * 3)
    histTop = max(max(r.materialTemperature()[c].max(), r.radiationTemperature()[c].max()) for r, c, _ in samples)
    for histAxis in axes[2]:
        histAxis.set_ylim(0, 1.05 * histTop)

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=TEMPERATURE_CMAP), ax=axes[0].tolist(), aspect=20)
    colorbar.set_label("Material $T$ at that time (keV)")
    colorbar = fig.colorbar(mesh, ax=axes[1].tolist(), aspect=20)
    colorbar.set_label(r"Energy share per $\ln\nu$")
    fig.suptitle(f"Spectrum as the Marshak front passes x = {position:.3g} cm", fontsize="large")
    if save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        boundaries = "_".join(dict.fromkeys(boundaryName(r).replace("/", "-") for r in results))
        path = Base.figurePath("spectra", f"FrontSpectrum_{boundaries}_x{position:.3g}_{stamp}.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Spectrum evolution as a Marshak wave front passes one cell.")
    parser.add_argument("files", nargs="*", help=".h5 result files (default: file dialog)")
    parser.add_argument("--set", dest="runSet", help="every non-MMS run in datastash/<RUNSET>/")
    parser.add_argument("--x", dest="position", type=float, help="cell position (cm)")
    parser.add_argument("--n", dest="count", type=int, default=8, help="number of spectra (default 8)")
    parser.add_argument("--planck", action="store_true", help="overlay the local Planckian")
    parser.add_argument("--emin", type=float, help="lowest photon energy shown (keV)")
    parser.add_argument("--emax", type=float, help="highest photon energy shown (keV)")
    parser.add_argument("--groups", type=int, help="with --set: only runs with this many groups")
    args = parser.parse_args()

    if args.files:
        paths = args.files
    elif args.runSet:
        paths = Results.findResults(args.runSet, mmsOnly=False)
    else:
        paths = Results.pickFiles("Select Marshak-wave results")
    if not paths:
        print("No results selected.")
        return
    results = Results.loadResults(paths)
    if args.groups is not None:
        results = [r for r in results if int(r.params["groups"]) == args.groups]
    if not results:
        print(f"No runs with {args.groups} groups.")
        return
    results.sort(key=lambda r: (int(r.params["groups"]), r.method))   # SM then VMG per group count
    plotFrontSpectra(results, args.position, args.count, args.planck, (args.emin, args.emax))


if __name__ == "__main__":
    main()
