"""Absolute error of saved MMS runs (MMSCompare = True) against the exact manufactured solution.

    python postprocessing/MMSError.py                   # pick .h5 files in a dialog
    python postprocessing/MMSError.py run1.h5 run2.h5   # explicit files
    python postprocessing/MMSError.py --all             # every MMS run under datastash/

Writes three files to figures/mmserror/ (or --out DIR), sharing one timestamped stem:
    MMSError_<stamp>.txt   plain-text error table
    MMSError_<stamp>.tex   LaTeX table (booktabs)
    MMSError_<stamp>.pdf   plots: error vs. time, error vs. x at the final time, and
                           (when a method has 2+ runs with different time steps) temporal convergence

Errors are absolute, in keV, for the material temperature T and the radiation temperature
T_r = (sum_g phi_g / (a c))^(1/4), compared with the exact MMS profiles
    T^4   = offset^4 + T0^4 (1 - e^{-tau t/2}) G(x)
    T_r^4 = offset^4 + T0^4 (1 + e^{-tau t/2}) G(x),   G = Gaussian of width sigma_x.
L2 is the cell-width-weighted RMS over x; Linf is the max over x.
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

import h5py
import numpy as np

import Base
from Base import Constants as const
import MMS
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
    with h5py.File(filepath, "r") as f:
        if not bool(f.attrs.get("MMSCompare", False)):
            return None
        data = {key: f[key][:] for key in ("fullTensorPhi", "temperatureSet", "timeSet")}
        dt = f["dt"][:]
        params = {
            "spaceGrid": f["spaceGrid"][:],
            "spaceMid": f["spaceMid"][:],
            "nSteps": int(f.attrs.get("nSteps", len(dt))),
            "groups": int(f.attrs.get("groups", data["fullTensorPhi"].shape[1])),
            "nBins": int(f.attrs.get("nBins", len(f["spaceMid"]))),
            "maxDt": float(np.max(dt)),
            "mmsT0": float(f.attrs["mmsT0"]),
            "mmsSigmaX": float(f.attrs["mmsSigmaX"]),
            "mmsTau": float(f.attrs["mmsTau"]),
            "mmsTemperatureOffset": float(
                f.attrs.get("mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET)
            ),
            "method": "VMG" if bool(f.attrs.get("movingCoordinates", False)) else "SM",
            "fileName": os.path.splitext(os.path.basename(filepath))[0],
        }
    return data, params


def findMMSFiles():
    paths = sorted(glob.glob(os.path.join(Base.DATA_DIR, "**", "*.h5"), recursive=True))
    return [p for p in paths if "failed" not in os.path.basename(os.path.dirname(p)).lower()]


def pickFiles():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select MMS HDF5 results",
        initialdir=Base.DATA_DIR,
        filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")],
    )
    root.destroy()
    return list(paths)


# --- Summary table -------------------------------------------------------------------------
def summarize(runs):
    """One row per run, sorted by method then time step, with observed temporal order."""
    rows = []
    for data, params, errors in runs:
        row = {
            "method": params["method"],
            "nSteps": params["nSteps"],
            "groups": params["groups"],
            "nBins": params["nBins"],
            "maxDt": params["maxDt"],
            "fileName": params["fileName"],
        }
        for name, _ in QUANTITIES:
            row[f"{name}_l2"] = errors[name]["l2"][-1]
            row[f"{name}_linf"] = errors[name]["linf"][-1]
            row[f"{name}_peak"] = np.max(errors[name]["linf"])
            row[f"{name}_order"] = np.nan
        rows.append(row)

    rows.sort(key=lambda r: (r["method"], r["groups"], r["nBins"], -r["maxDt"]))
    for previous, current in zip(rows, rows[1:]):
        sameFamily = all(previous[k] == current[k] for k in ("method", "groups", "nBins"))
        if not sameFamily or previous["maxDt"] == current["maxDt"]:
            continue
        for name, _ in QUANTITIES:
            ratio = previous[f"{name}_l2"] / current[f"{name}_l2"]
            if ratio > 0 and np.isfinite(ratio):
                order = np.log(ratio) / np.log(previous["maxDt"] / current["maxDt"])
                current[f"{name}_order"] = round(order, 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    return rows


def writeText(rows, path):
    header = (
        f"{'Method':<6} {'Steps':>6} {'Groups':>6} {'Bins':>5} {'max dt':>10} | "
        f"{'T L2':>10} {'T Linf':>10} {'T peak':>10} {'T p':>6} | "
        f"{'Tr L2':>10} {'Tr Linf':>10} {'Tr peak':>10} {'Tr p':>6}  File"
    )
    lines = [
        "Absolute MMS error (keV). L2 and Linf over x at the final time; peak = max over all",
        "times of Linf; p = observed temporal order from the L2 error vs. the previous row.",
        "",
        header,
        "-" * len(header),
    ]
    for r in rows:
        cells = [f"{r['method']:<6} {r['nSteps']:>6d} {r['groups']:>6d} {r['nBins']:>5d} {r['maxDt']:>10.3e}"]
        for name, _ in QUANTITIES:
            order = r[f"{name}_order"]
            cells.append(
                f"{r[f'{name}_l2']:>10.3e} {r[f'{name}_linf']:>10.3e} {r[f'{name}_peak']:>10.3e} "
                + (f"{order:>6.2f}" if np.isfinite(order) else f"{'--':>6}")
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
        "$p$ is the observed temporal order of the $L_2$ error.}",
        "  \\label{tab:mms-error}",
        "  \\begin{tabular}{lrrr cccc cccc}",
        "    \\toprule",
        "    & & & & \\multicolumn{4}{c}{Material $T$} & \\multicolumn{4}{c}{Radiation $T_r$} \\\\",
        "    \\cmidrule(lr){5-8} \\cmidrule(lr){9-12}",
        "    Method & Steps & Groups & $\\Delta t_{\\max}$ "
        "& $L_2$ & $L_\\infty$ & peak & $p$ & $L_2$ & $L_\\infty$ & peak & $p$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        cells = [r["method"], str(r["nSteps"]), str(r["groups"]), latexNumber(r["maxDt"])]
        for name, _ in QUANTITIES:
            order = r[f"{name}_order"]
            cells += [
                latexNumber(r[f"{name}_l2"]),
                latexNumber(r[f"{name}_linf"]),
                latexNumber(r[f"{name}_peak"]),
                f"{order:.2f}" if np.isfinite(order) else "--",
            ]
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Plots ---------------------------------------------------------------------------------
def runLabel(params):
    return f"{params['method']} {params['nSteps']} steps, {params['groups']} groups"


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

        # Page 3: temporal convergence of the final-time L2 error, one line per method family
        families = {}
        for r in rows:
            families.setdefault((r["method"], r["groups"], r["nBins"]), []).append(r)
        families = {k: v for k, v in families.items() if len({r["maxDt"] for r in v}) > 1}
        if families:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for i, ((method, groups, nBins), family) in enumerate(families.items()):
                dt = np.array([r["maxDt"] for r in family])
                for axis, (name, _) in zip(axes, QUANTITIES):
                    error = np.array([r[f"{name}_l2"] for r in family])
                    slope = round(np.polyfit(np.log(dt), np.log(error), 1)[0], 2) + 0.0
                    axis.loglog(dt, error, "o-", color=colors[i % len(colors)],
                                label=f"{method}, {groups} groups, {nBins} bins (fit p = {slope:.2f})")
            for axis, (name, symbol) in zip(axes, QUANTITIES):
                dtRange = np.array(axis.get_xlim())
                anchorDt = dtRange[1]
                anchorError = min(line.get_ydata()[np.argmax(line.get_xdata())] for line in axis.get_lines())
                axis.loglog(dtRange, anchorError * (dtRange / anchorDt), color="gray",
                            linestyle=":", label="first order")
                axis.set_title(f"{name.capitalize()} temperature ${symbol}$")
                axis.set_xlabel(r"$\Delta t_{\max}$ (ns)")
                axis.set_ylabel(r"Final-time $L_2$ error (keV)")
                styleAxis(axis)
                axis.legend(frameon=False, fontsize="small")
            fig.suptitle("MMS temporal convergence")
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

    runs.sort(key=lambda run: (run[1]["method"], run[1]["groups"], -run[1]["maxDt"]))
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
    parser.add_argument("--out", default=OUTPUT_DIR, help="output directory (default: figures/mmserror/)")
    args = parser.parse_args()

    filepaths = args.files or (findMMSFiles() if args.all else pickFiles())
    if not filepaths:
        print("No files selected.")
        return
    analyze(filepaths, args.out)


if __name__ == "__main__":
    main()
