"""Absolute error of saved MMS runs (MMSCompare = True) against the exact manufactured solution.

    python postprocessing/MMSError.py                   # pick .h5 files in a dialog
    python postprocessing/MMSError.py run1.h5 run2.h5   # explicit files
    python postprocessing/MMSError.py --all             # every MMS run under datastash/
    python postprocessing/MMSError.py --set MMSRuns     # every MMS run in datastash/MMSRuns/

Writes three files to figures/mmserror/ (or --out DIR), sharing one timestamped stem:
    MMSError_<stamp>.txt   plain-text error tables
    MMSError_<stamp>.tex   LaTeX tables (booktabs)
    MMSError_<stamp>.pdf   plots: error vs. time, error vs. x at the final time, then one page per
                           refinement study (SM dotted with markers, VMG dashed)

There are three refinement studies, each with its own table and plot: dx varies (at the most time
steps and groups), dt varies (at Nx=100, G=100) and the group count varies (at Nx=100, Nt=2000).
The fixed values are set in STUDY_FIXED; any not set there use the finest available.

Errors are absolute, in keV, for the material temperature T, compared with the exact MMS profile
    T^4 = offset^4 + T0^4 (1 - e^{-tau t/2}) G(x),   G = Gaussian of width sigma_x.
The radiation temperature T_r = (sum_g phi_g / (a c))^(1/4) (exact: 1 + e^{-tau t/2} in place of
1 - e^{-tau t/2}) is also computed; add it to QUANTITIES to report it in the plots.
L2 is the cell-width-weighted RMS over x; Linf is the max over x, both at the final time.
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
import MMS
import Results
import Visualize  # noqa: F401  (applies the project's matplotlib style)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, NullFormatter


QUANTITIES = (("material", "T"),)   # add ("radiation", "T_r") to also report T_r
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


# --- Refinement studies -------------------------------------------------------------------
# Each study varies one resolution and holds the other two fixed. The values held fixed are set
# here (nBins = Nx, nSteps = Nt, groups = G); any not given use the finest available for each
# method (most cells, most time steps = smallest dt, most groups).
STUDY_FIXED = {
    "dx": {},                                   # finest Nt and G
    "dt": {"nBins": 100, "groups": 100},
    "groups": {"nBins": 100, "nSteps": 2000},
}

STUDIES = (
    dict(key="dx", vary="nBins", axis="dx", title="Spatial refinement",
         columns=(("Nx", "nBins"), ("dx", "dx")),
         texColumns=(("$N_x$", "nBins"), ("$\\Delta x$ (cm)", "dx")),
         xLabel=r"$\Delta x$ (cm)", reference=1),
    dict(key="dt", vary="nSteps", axis="maxDt", title="Temporal refinement",
         columns=(("Nt", "nSteps"), ("max dt", "maxDt")),
         texColumns=(("$N_t$", "nSteps"), ("$\\Delta t_{\\max}$ (ns)", "maxDt")),
         xLabel=r"$\Delta t_{\max}$ (ns)", reference=1),
    dict(key="groups", vary="groups", axis="groups", title="Group refinement",
         columns=(("G", "groups"),), texColumns=(("$G$", "groups"),),
         xLabel="Groups $G$", reference=None),
)
RESOLUTION_KEYS = ("nBins", "nSteps", "groups")
FIXED_TEXT = {"nBins": lambda r: f"Nx={r['nBins']} (dx={r['dx']:.3g} cm)",
              "nSteps": lambda r: f"Nt={r['nSteps']} (max dt={r['maxDt']:.3g} ns)",
              "groups": lambda r: f"G={r['groups']}"}
FIXED_TEX = {"nBins": lambda r: f"$\\Delta x = {r['dx']:.3g}$~cm",
             "nSteps": lambda r: f"$\\Delta t_{{\\max}} = {r['maxDt']:.3g}$~ns",
             "groups": lambda r: f"$G = {r['groups']}$"}


def summarize(runs):
    """One row per run with its final-time L2 and Linf errors. Repeated runs (same method and
    resolution) keep the newest file."""
    rows = {}
    for data, params, errors in runs:
        row = {
            "method": params["method"],
            "nSteps": int(params["nSteps"]),
            "groups": int(params["groups"]),
            "nBins": int(params["nBins"]),
            "dx": float(params["dx"]),
            "maxDt": float(params["maxDt"]),
            "runTime": float(params.get("runTime", np.nan)),   # wall-clock solve time (s)
            "fileName": params["fileName"],
        }
        for name, _ in QUANTITIES:
            row[f"{name}_l2"] = errors[name]["l2"][-1]
            row[f"{name}_linf"] = errors[name]["linf"][-1]
        key = (row["method"],) + tuple(row[k] for k in RESOLUTION_KEYS)
        if key not in rows or row["fileName"] > rows[key]["fileName"]:
            rows[key] = row
    return list(rows.values())


def buildStudies(rows):
    """{study key: (study, {method: (row with the fixed values, [rows coarse -> fine])})}."""
    studies = {}
    for study in STUDIES:
        fixedKeys = [k for k in RESOLUTION_KEYS if k != study["vary"]]
        chosen = STUDY_FIXED.get(study["key"], {})
        byMethod = {}
        for method in sorted({r["method"] for r in rows}):
            mine = [r for r in rows if r["method"] == method]
            fixed = {k: chosen.get(k, max(r[k] for r in mine)) for k in fixedKeys}
            members = sorted((r for r in mine if all(r[k] == fixed[k] for k in fixedKeys)),
                             key=lambda r: r[study["vary"]])
            if members:
                byMethod[method] = (members[-1], members)
            else:
                print(f"{study['title']}: no {method} runs with "
                      + ", ".join(f"{k}={v}" for k, v in fixed.items()))
        studies[study["key"]] = (study, byMethod)
    return studies


def fixedDescription(study, byMethod, formats, joiner="; "):
    """The values held fixed in a study; listed per method only when the methods differ."""
    fixedKeys = [k for k in RESOLUTION_KEYS if k != study["vary"]]
    texts = {m: ", ".join(formats[k](fixed) for k in fixedKeys) for m, (fixed, _) in byMethod.items()}
    if len(set(texts.values())) == 1:
        return next(iter(texts.values()))
    return joiner.join(f"{m}: {t}" for m, t in texts.items())


def percentChange(previous, row, key):
    """Signed % change of row[key] from the previous (coarser) row; nan for the first row."""
    if previous is None or not (np.isfinite(previous[key]) and previous[key] != 0):
        return np.nan
    return round(100.0 * (row[key] - previous[key]) / previous[key], 1) + 0.0   # + 0.0: no "-0.0%"


def _percentText(value, width):
    return f"{value:>+{width}.1f}%" if np.isfinite(value) else f"{'--':>{width + 1}}"


def _percentTex(value):
    return f"${value:+.1f}\\%$" if np.isfinite(value) else "--"


def _textValue(value):
    return f"{value:>10.3e}" if isinstance(value, float) else f"{value:>10d}"


def writeText(studies, path):
    lines = ["Absolute material-temperature MMS error (keV): L2 and Linf over x at the final time;",
             "runtime = wall-clock solve time (s). Each table varies one resolution with the other two",
             "fixed (values in brackets). chg = signed % change (L2 error, runtime) from the previous",
             "(coarser) row of the same method; negative = decrease.",
             ""]
    for study, byMethod in studies.values():
        if not byMethod:
            continue
        header = (f"{'Method':<6} " + " ".join(f"{label:>10}" for label, _ in study["columns"])
                  + f" | {'T L2':>10} {'T Linf':>10} {'chg':>8} | {'Runtime (s)':>11} {'chg':>8}")
        lines += [f"{study['title']}  [{fixedDescription(study, byMethod, FIXED_TEXT)}]",
                  header, "-" * len(header)]
        for method, (_, members) in byMethod.items():
            previous = None
            for r in members:
                cells = [f"{method:<6} " + " ".join(_textValue(r[k]) for _, k in study["columns"])]
                cells.append(f"{r['material_l2']:>10.3e} {r['material_linf']:>10.3e} "
                             + _percentText(percentChange(previous, r, "material_l2"), 7))
                runTime = f"{r['runTime']:>11.1f}" if np.isfinite(r["runTime"]) else f"{'--':>11}"
                cells.append(runTime + " " + _percentText(percentChange(previous, r, "runTime"), 7))
                lines.append(" | ".join(cells))
                previous = r
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def latexNumber(value):
    if not np.isfinite(value):
        return "--"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def writeLatex(studies, path):
    lines = ["% Requires \\usepackage{booktabs}"]
    for study, byMethod in studies.values():
        if not byMethod:
            continue
        nLead = 1 + len(study["texColumns"])
        lines += [
            "",
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{study['title']}: absolute MMS error of the material temperature (keV) at "
            f"the final time and wall-clock runtime, with {fixedDescription(study, byMethod, FIXED_TEX)}. "
            "Change columns give the signed percentage change of the $L_2$ error and of the runtime "
            "from the previous (coarser) row of the same method.}",
            f"  \\label{{tab:mms-error-{study['key']}}}",
            f"  \\begin{{tabular}}{{l{'r' * (nLead - 1)} ccr rr}}",
            "    \\toprule",
            "    " + "& " * (nLead - 1)
            + "& \\multicolumn{3}{c}{Material $T$ error} & \\multicolumn{2}{c}{Runtime} \\\\",
            f"    \\cmidrule(lr){{{nLead + 1}-{nLead + 3}}} \\cmidrule(lr){{{nLead + 4}-{nLead + 5}}}",
            "    Method & " + " & ".join(label for label, _ in study["texColumns"])
            + " & $L_2$ & $L_\\infty$ & Change & (s) & Change \\\\",
            "    \\midrule",
        ]
        for i, (method, (_, members)) in enumerate(byMethod.items()):
            if i:
                lines.append("    \\addlinespace")
            previous = None
            for r in members:
                cells = [method] + [latexNumber(r[k]) if isinstance(r[k], float) else str(r[k])
                                    for _, k in study["texColumns"]]
                cells += [latexNumber(r["material_l2"]), latexNumber(r["material_linf"]),
                          _percentTex(percentChange(previous, r, "material_l2"))]
                cells.append(f"{r['runTime']:.1f}" if np.isfinite(r["runTime"]) else "--")
                cells.append(_percentTex(percentChange(previous, r, "runTime")))
                lines.append("    " + " & ".join(cells) + " \\\\")
                previous = r
        lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Plots ---------------------------------------------------------------------------------
def runLabel(params):
    return params["label"]


def styleAxis(axis):
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)


def writePdf(runs, studies, path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    with PdfPages(path) as pdf:
        # Page 1: Linf (solid) and L2 (dashed) error vs. time
        fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(6.5 * len(QUANTITIES), 4.5), sharey=True, squeeze=False)
        axes = axes[0]
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
        axes[-1].plot([], [], color="gray", label=r"$L_\infty$")
        axes[-1].plot([], [], color="gray", linestyle="--", label=r"$L_2$")
        axes[-1].legend(frameon=False, fontsize="small")
        fig.suptitle("MMS absolute error vs. time")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: error profile in x at the final time
        fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(6.5 * len(QUANTITIES), 4.5), sharey=True, squeeze=False)
        axes = axes[0]
        for i, (_, params, errors) in enumerate(runs):
            for axis, (name, _) in zip(axes, QUANTITIES):
                axis.semilogy(errors["x"], errors[name]["error"][-1],
                              color=colors[i % len(colors)], label=runLabel(params))
        for axis, (name, symbol) in zip(axes, QUANTITIES):
            axis.set_title(f"{name.capitalize()} temperature ${symbol}$")
            axis.set_xlabel("x (cm)")
            styleAxis(axis)
        axes[0].set_ylabel("Absolute error (keV)")
        axes[-1].legend(frameon=False, fontsize="small")
        fig.suptitle(f"MMS absolute error at the final time, t = {runs[0][2]['time'][-1]:g} ns")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 3-5: one per refinement study (dx, dt, groups)
        for study, byMethod in studies.values():
            studyPage(pdf, study, byMethod, colors)


METHOD_STYLE = {"SM": dict(linestyle=":", linewidth=1.8, marker="o", markersize=4), "VMG": dict(linestyle="--")}
NORMS = (("l2", r"$L_2$"), ("linf", r"$L_\infty$"))


def studyPage(pdf, study, byMethod, colors):
    """Final-time L2 and Linf error vs. the varied resolution: SM dotted with markers, VMG dashed."""
    if not byMethod:
        return
    fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(6.5 * len(QUANTITIES), 4.5), squeeze=False)
    axes = axes[0]
    for axis, (name, symbol) in zip(axes, QUANTITIES):
        for method, (_, members) in byMethod.items():
            x = np.array([r[study["axis"]] for r in members], dtype=float)
            for color, (norm, normLabel) in zip(colors, NORMS):
                axis.loglog(x, [r[f"{name}_{norm}"] for r in members], color=color,
                            label=f"{method} {normLabel}", **METHOD_STYLE.get(method, {}))
        allX = np.array([r[study["axis"]] for _, ms in byMethod.values() for r in ms], dtype=float)
        if study["reference"] and len(np.unique(allX)) > 1:
            xs = np.array([allX.min(), allX.max()])
            anchor = min(ms[-1][f"{name}_l2"] for _, ms in byMethod.values())
            axis.loglog(xs, anchor * (xs / xs[1]) ** study["reference"], color="gray",
                        linestyle="-.", linewidth=1, label="first order")
        axis.set_title(f"{name.capitalize()} temperature ${symbol}$")
        axis.set_xlabel(study["xLabel"])
        axis.set_ylabel("Final-time absolute error (keV)")
        styleAxis(axis)
        axis.xaxis.set_minor_formatter(NullFormatter())
        if study["key"] == "groups":   # label the group counts that were run
            axis.set_xticks(np.unique(allX))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        axis.legend(frameon=False, fontsize="small")
    fig.suptitle(f"MMS {study['title'].lower()}\n{fixedDescription(study, byMethod, FIXED_TEXT)}",
                 fontsize="medium")
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
    studies = buildStudies(summarize(runs))
    os.makedirs(outDir, exist_ok=True)
    stem = os.path.join(outDir, f"MMSError_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writeText(studies, stem + ".txt")
    writeLatex(studies, stem + ".tex")
    writePdf(runs, studies, stem + ".pdf")
    with open(stem + ".txt", encoding="utf-8") as f:
        print(f.read())
    print(f"Wrote {stem}.txt, .tex and .pdf")
    return studies


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
