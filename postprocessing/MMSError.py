"""Absolute error of saved MMS runs (MMSCompare = True) against the exact manufactured solution.

    python postprocessing/MMSError.py                   # pick .h5 files in a dialog
    python postprocessing/MMSError.py run1.h5 run2.h5   # explicit files
    python postprocessing/MMSError.py --all             # every MMS run under datastash/
    python postprocessing/MMSError.py --set MMSRuns     # every MMS run in datastash/MMSRuns/

Writes three files to figures/mmserror/ (or --out DIR), sharing one timestamped stem:
    MMSError_<stamp>.txt   plain-text error table
    MMSError_<stamp>.tex   LaTeX table (booktabs)
    MMSError_<stamp>.pdf   plots: error vs. time, error vs. x at the final time, temporal
                           convergence (runs differing only in dt) and spatial convergence
                           (runs differing only in dx)

Errors are absolute, in keV, for the material temperature T and the radiation temperature
T_r = (sum_g phi_g / (a c))^(1/4), compared with the exact MMS profiles
    T^4   = offset^4 + T0^4 (1 - e^{-tau t/2}) G(x)
    T_r^4 = offset^4 + T0^4 (1 + e^{-tau t/2}) G(x),   G = Gaussian of width sigma_x.
L2 is the cell-width-weighted RMS over x; Linf is the max over x. Observed orders p_t and p_x
compare a run with the next coarser run of the same family (same method and groups, and the same
dx for p_t or the same time steps for p_x).
"""
import argparse
import glob
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
import MMS
import Results
import Visualize  # noqa: F401  (applies the project's matplotlib style)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


QUANTITIES = (("material", "T"), ("radiation", "T_r"))
OUTPUT_DIR = os.path.join(Base.FIGURE_DIR, "mmserror")


# --- Exact solution and errors -------------------------------------------------------------
def exactTemperatures(params, time, x):
    """Exact (material, radiation) temperatures, each shaped (nTime, nBins)."""
    sigmaX = params["mmsSigmaX"]
    offset = params.get("mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET)
    spatial = np.exp(-(x**2) / (2 * sigmaX**2)) / (sigmaX * np.sqrt(2 * np.pi))
    decay = np.exp(-params["mmsTau"] * time / 2)[:, None]
    material = MMS.temperatureFromFourthPowerProfile(
        params["mmsT0"], (1 - decay) * spatial[None, :], offset
    )
    radiation = MMS.temperatureFromFourthPowerProfile(
        params["mmsT0"], (1 + decay) * spatial[None, :], offset
    )
    return material, radiation


def runErrors(data, params):
    """Absolute error histories for one run (time axis first)."""
    timeCount = min(
        len(data["timeSet"]),
        data["temperatureSet"].shape[1],
        data["fullTensorPhi"].shape[0],
    )
    time = np.asarray(data["timeSet"])[:timeCount]
    x = np.asarray(params["spaceMid"])
    widths = np.diff(np.asarray(params["spaceGrid"]))

    material = data["temperatureSet"][:, :timeCount].T
    energy = np.sum(data["fullTensorPhi"][:timeCount], axis=1) / (const.a * const.c)
    radiation = np.maximum(energy, 0.0) ** 0.25
    exactMaterial, exactRadiation = exactTemperatures(params, time, x)

    result = {"time": time, "x": x}
    for name, numeric, exact in (
        ("material", material, exactMaterial),
        ("radiation", radiation, exactRadiation),
    ):
        error = np.abs(numeric - exact)
        result[name] = {
            "error": error,                                     # (nTime, nBins)
            "linf": np.max(error, axis=1),                      # (nTime,)
            "l2": np.sqrt(np.sum(widths * error**2, axis=1) / np.sum(widths)),
        }
    return result


# --- Loading -------------------------------------------------------------------------------
def loadRun(filepath):
    """(data, params) of an MMS run, or None for a non-MMS run."""
    result = Results.loadResult(filepath)
    if not result.params["MMSCompare"]:
        return None
    params = dict(result.params)
    params["fileName"] = result.fileName
    params["label"] = result.label
    return result.data, params


def findMMSFiles(runSet=None):
    return Results.findResults(runSet, mmsOnly=True)


def pickFiles():
    return Results.pickFiles("Select MMS HDF5 results")


# --- Summary table -------------------------------------------------------------------------
def _observedOrders(rows, familyKeys, stepKey, suffix):
    """Order of the final-time L2 error vs. the next coarser run (larger stepKey) in each family."""
    families = {}
    for row in rows:
        families.setdefault(tuple(row[k] for k in familyKeys), []).append(row)
    for family in families.values():
        family.sort(key=lambda r: -r[stepKey])
        for coarse, fine in zip(family, family[1:]):
            if not fine[stepKey] < coarse[stepKey]:
                continue
            for name, _ in QUANTITIES:
                ratio = coarse[f"{name}_l2"] / fine[f"{name}_l2"]
                if ratio > 0 and np.isfinite(ratio):
                    order = np.log(ratio) / np.log(coarse[stepKey] / fine[stepKey])
                    fine[f"{name}_{suffix}"] = round(order, 2) + 0.0  # + 0.0 turns -0.0 into 0.0


def summarize(runs):
    """One row per run, sorted by method, groups, dx then dt, with observed orders in t and x."""
    rows = []
    for data, params, errors in runs:
        row = {
            "method": params["method"],
            "nSteps": int(params["nSteps"]),
            "groups": int(params["groups"]),
            "nBins": int(params["nBins"]),
            "dx": float(params["dx"]),
            "maxDt": float(params["maxDt"]),
            "fileName": params["fileName"],
        }
        for name, _ in QUANTITIES:
            row[f"{name}_l2"] = errors[name]["l2"][-1]
            row[f"{name}_linf"] = errors[name]["linf"][-1]
            row[f"{name}_peak"] = np.max(errors[name]["linf"])
            row[f"{name}_pt"] = np.nan
            row[f"{name}_px"] = np.nan
        rows.append(row)

    _observedOrders(rows, ("method", "groups", "dx"), "maxDt", "pt")
    _observedOrders(rows, ("method", "groups", "nSteps"), "dx", "px")
    rows.sort(key=lambda r: (r["method"], r["groups"], -r["dx"], -r["maxDt"]))
    return rows


def _orderCell(value, width):
    return f"{value:>{width}.2f}" if np.isfinite(value) else f"{'--':>{width}}"


def writeText(rows, path):
    header = (
        f"{'Method':<6} {'Groups':>6} {'Bins':>5} {'dx':>9} {'Steps':>6} {'max dt':>10} | "
        f"{'T L2':>10} {'T Linf':>10} {'T peak':>10} {'T p_t':>6} {'T p_x':>6} | "
        f"{'Tr L2':>10} {'Tr Linf':>10} {'Tr peak':>10} {'Tr p_t':>6} {'Tr p_x':>6}  File"
    )
    lines = [
        "Absolute MMS error (keV). L2 and Linf over x at the final time; peak = max over all",
        "times of Linf; p_t / p_x = observed order of the L2 error in dt / dx (see module docstring).",
        "",
        header,
        "-" * len(header),
    ]
    for r in rows:
        cells = [f"{r['method']:<6} {r['groups']:>6d} {r['nBins']:>5d} {r['dx']:>9.3e} "
                 f"{r['nSteps']:>6d} {r['maxDt']:>10.3e}"]
        for name, _ in QUANTITIES:
            cells.append(
                f"{r[f'{name}_l2']:>10.3e} {r[f'{name}_linf']:>10.3e} {r[f'{name}_peak']:>10.3e} "
                + _orderCell(r[f"{name}_pt"], 6) + " " + _orderCell(r[f"{name}_px"], 6)
            )
        lines.append(" | ".join(cells) + f"  {r['fileName']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def latexNumber(value):
    if not np.isfinite(value):
        return "--"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def writeLatex(rows, path):
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Absolute error of the MMS runs against the manufactured solution (keV). "
        "$L_2$ and $L_\\infty$ are over $x$ at the final time; peak is $\\max_t L_\\infty$; "
        "$p_t$ and $p_x$ are the observed orders of the $L_2$ error in $\\Delta t$ and $\\Delta x$.}",
        "  \\label{tab:mms-error}",
        "  \\begin{tabular}{lrrr ccccc ccccc}",
        "    \\toprule",
        "    & & & & \\multicolumn{5}{c}{Material $T$} & \\multicolumn{5}{c}{Radiation $T_r$} \\\\",
        "    \\cmidrule(lr){5-9} \\cmidrule(lr){10-14}",
        "    Method & Groups & $\\Delta x$ & $\\Delta t_{\\max}$ "
        "& $L_2$ & $L_\\infty$ & peak & $p_t$ & $p_x$ & $L_2$ & $L_\\infty$ & peak & $p_t$ & $p_x$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        cells = [r["method"], str(r["groups"]), latexNumber(r["dx"]), latexNumber(r["maxDt"])]
        for name, _ in QUANTITIES:
            cells += [latexNumber(r[f"{name}_l2"]), latexNumber(r[f"{name}_linf"]), latexNumber(r[f"{name}_peak"])]
            cells += [f"{r[f'{name}_{k}']:.2f}" if np.isfinite(r[f"{name}_{k}"]) else "--" for k in ("pt", "px")]
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Plots ---------------------------------------------------------------------------------
def runLabel(params):
    return params["label"]


def styleAxis(axis):
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)


def writePdf(runs, rows, path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    with PdfPages(path) as pdf:
        # Page 1: Linf (solid) and L2 (dashed) error vs. time
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for i, (_, params, errors) in enumerate(runs):
            color = colors[i % len(colors)]
            for axis, (name, _) in zip(axes, QUANTITIES):
                t = errors["time"][1:]
                axis.semilogy(t, errors[name]["linf"][1:], color=color, label=runLabel(params))
                axis.semilogy(t, errors[name]["l2"][1:], color=color, linestyle="--")
        for axis, (name, symbol) in zip(axes, QUANTITIES):
            axis.set_title(f"{name.capitalize()} temperature ${symbol}$")
            axis.set_xlabel("Time (ns)")
            styleAxis(axis)
        axes[0].set_ylabel("Absolute error (keV)")
        axes[1].plot([], [], color="gray", label=r"$L_\infty$")
        axes[1].plot([], [], color="gray", linestyle="--", label=r"$L_2$")
        axes[1].legend(frameon=False, fontsize="small")
        fig.suptitle("MMS absolute error vs. time")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: error profile in x at the final time
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for i, (_, params, errors) in enumerate(runs):
            for axis, (name, _) in zip(axes, QUANTITIES):
                axis.semilogy(errors["x"], errors[name]["error"][-1],
                              color=colors[i % len(colors)], label=runLabel(params))
        for axis, (name, symbol) in zip(axes, QUANTITIES):
            axis.set_title(f"{name.capitalize()} temperature ${symbol}$")
            axis.set_xlabel("x (cm)")
            styleAxis(axis)
        axes[0].set_ylabel("Absolute error (keV)")
        axes[1].legend(frameon=False, fontsize="small")
        fig.suptitle(f"MMS absolute error at the final time, t = {runs[0][2]['time'][-1]:g} ns")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: temporal convergence (same method, groups, dx); page 4: spatial (same method, groups, steps)
        convergencePage(pdf, rows, colors, ("method", "groups", "dx"), "maxDt",
                        r"$\Delta t_{\max}$ (ns)", "MMS temporal convergence",
                        lambda m, g, dx: f"{m}, G={g}, Δx={dx:.3g}")
        convergencePage(pdf, rows, colors, ("method", "groups", "nSteps"), "dx",
                        r"$\Delta x$ (cm)", "MMS spatial convergence",
                        lambda m, g, n: f"{m}, G={g}, Nt={n}")


def convergencePage(pdf, rows, colors, familyKeys, stepKey, xLabel, title, familyLabel):
    """Final-time L2 error vs. step size for each family with 2+ step sizes, with a fitted order
    and a first-order reference line."""
    families = {}
    for r in rows:
        families.setdefault(tuple(r[k] for k in familyKeys), []).append(r)
    families = {k: sorted(v, key=lambda r: r[stepKey]) for k, v in families.items()
                if len({r[stepKey] for r in v}) > 1}
    if not families:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (key, family) in enumerate(families.items()):
        step = np.array([r[stepKey] for r in family])
        for axis, (name, _) in zip(axes, QUANTITIES):
            error = np.array([r[f"{name}_l2"] for r in family])
            slope = round(np.polyfit(np.log(step), np.log(error), 1)[0], 2) + 0.0
            axis.loglog(step, error, "o-", color=colors[i % len(colors)],
                        label=f"{familyLabel(*key)} (fit p = {slope:.2f})")
    for axis, (name, symbol) in zip(axes, QUANTITIES):
        stepRange = np.array(axis.get_xlim())
        anchorError = min(line.get_ydata()[np.argmax(line.get_xdata())] for line in axis.get_lines())
        axis.loglog(stepRange, anchorError * (stepRange / stepRange[1]), color="gray",
                    linestyle=":", label="first order")
        axis.set_title(f"{name.capitalize()} temperature ${symbol}$")
        axis.set_xlabel(xLabel)
        axis.set_ylabel(r"Final-time $L_2$ error (keV)")
        styleAxis(axis)
        axis.legend(frameon=False, fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# --- Entry point ---------------------------------------------------------------------------
def analyze(filepaths, outDir=OUTPUT_DIR):
    runs = []
    for path in filepaths:
        loaded = loadRun(path)
        if loaded is None:
            print(f"Skipping (not an MMS run): {path}")
            continue
        data, params = loaded
        runs.append((data, params, runErrors(data, params)))
    if not runs:
        print("No MMS runs to analyze.")
        return None

    runs.sort(key=lambda run: (run[1]["method"], run[1]["groups"], -run[1]["dx"], -run[1]["maxDt"]))
    rows = summarize(runs)
    os.makedirs(outDir, exist_ok=True)
    stem = os.path.join(outDir, f"MMSError_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writeText(rows, stem + ".txt")
    writeLatex(rows, stem + ".tex")
    writePdf(runs, rows, stem + ".pdf")
    with open(stem + ".txt", encoding="utf-8") as f:
        print(f.read())
    print(f"Wrote {stem}.txt, .tex and .pdf")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Absolute error of saved MMS runs.")
    parser.add_argument("files", nargs="*", help="MMS .h5 result files (default: file dialog)")
    parser.add_argument("--all", action="store_true", help="use every MMS run under datastash/")
    parser.add_argument("--set", dest="runSet", help="use every MMS run in datastash/<RUNSET>/")
    parser.add_argument("--out", default=OUTPUT_DIR, help="output directory (default: figures/mmserror/)")
    args = parser.parse_args()

    if args.files:
        filepaths = args.files
    elif args.all or args.runSet:
        filepaths = findMMSFiles(args.runSet)
    else:
        filepaths = pickFiles()
    if not filepaths:
        print("No files selected.")
        return
    analyze(filepaths, args.out)


if __name__ == "__main__":
    main()
