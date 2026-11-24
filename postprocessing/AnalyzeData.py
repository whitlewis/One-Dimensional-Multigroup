"""Standard comparison plots for saved SM/VMG runs.

    python postprocessing/AnalyzeData.py                      # pick .h5 files in a dialog
    python postprocessing/AnalyzeData.py run1.h5 run2.h5      # explicit files
    python postprocessing/AnalyzeData.py --set MMSRuns        # every run in datastash/MMSRuns/

Runs are labelled "<method> · G=<groups> · Nt=<steps> · Δx=<cell width>" in every legend.
For MMS error tables and convergence plots use MMSError.py.
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("src", "postprocessing"):
    if os.path.join(_ROOT, _folder) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _folder))

import numpy as np

import Results
import Visualize as Vis


def plotResults(results, position=None):
    """MMS runs: snapshots vs. the exact MMS, one-cell temperature and its error.
    Other runs: snapshots, one-cell temperature, space-time map (single run) and spectra.

    position (cm) selects the cell for the one-cell plots (default: as in Visualize); each run
    uses its own cell nearest that x, so runs with different dx are compared at the same place.
    """
    mms = [r for r in results if r.params["MMSCompare"]]
    other = [r for r in results if not r.params["MMSCompare"]]

    if mms:
        Vis.plotMMSComparisonsLoaded(mms)
        Vis.plotTemperatureMMSLoaded(mms, position=position)
        Vis.plotMaterialTemperatureMMSErrorLoaded(mms, position=position)

    if other:
        Vis.plotTemperatureSnapshotsLoaded(other)
        Vis.plotTemperatureLoaded(other, position=position)
        if len(other) == 1:
            Vis.plotTemperatureTimeLoaded(other)
        timeIndices = np.floor(np.geomspace(1, other[0].timeCount - 2, 4)).astype(int)
        Vis.plot_spectra_at_times(other, timeIndices, position=position)


def plotSet(paths=None, runSet=None, position=None):
    """Load results (explicit paths, a whole run set, or a file dialog) and plot them."""
    if paths is None:
        paths = Results.findResults(runSet) if runSet else Results.pickFiles()
    if not paths:
        print("No results selected.")
        return []
    results = Results.loadResults(paths)
    plotResults(results, position=position)
    return results


def main():
    parser = argparse.ArgumentParser(description="Comparison plots for saved SM/VMG runs.")
    parser.add_argument("files", nargs="*", help=".h5 result files (default: file dialog)")
    parser.add_argument("--set", dest="runSet", help="plot every run in datastash/<RUNSET>/")
    parser.add_argument("--x", dest="position", type=float, help="x (cm) of the cell for one-cell plots")
    args = parser.parse_args()
    plotSet(args.files or None, args.runSet, args.position)


if __name__ == "__main__":
    main()
