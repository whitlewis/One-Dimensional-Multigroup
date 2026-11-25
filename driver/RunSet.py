"""Batch driver: run every combination of a parameter sweep with SM and/or VMG, in parallel.

    python driver/RunSet.py --both     # SM and VMG
    python driver/RunSet.py --sm       # or --vmg
    python driver/RunSet.py            # asks which method(s)  (a bare "both"/"sm"/"vmg" also works)

Edit RUN_SET, BASE and SWEEP below. Keys are Multigroup.Parameters attribute names. Each run is
saved to datastash/<RUN_SET>/<sm|vmg>/<runName>_<timestamp>.h5, where runName lists the swept
values (e.g. G100_Nt4000_Nx200_tol1e-14), and every parameter is stored in the file.
Analyse with postprocessing/AnalyzeData.py --set <RUN_SET> or MMSError.py --set <RUN_SET>.
"""
import argparse
import concurrent.futures
import itertools
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("src", "postprocessing"):
    if os.path.join(_ROOT, _folder) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _folder))

import Base
import MMS
import Multigroup as MG

# --- Run set -------------------------------------------------------------------------------------
RUN_SET = "BimodalMarshak"   # folder under datastash/

BASE = dict(
    runLabel="BimodalMarshak",
    # Space and angle
    xMin=-1, xMax=1, nBins=50, sn=8,
    # Frequency groups
    freqNum=100, groupSpace="log", minFreq=1e-4, maxFreq=45, infFreq=125,
    matchNu=True,                     # VMG: u edges match the SM edges at the initial temperature
    # Time stepping
    nSteps=1000, timeMax=0.5, timeScale="log",
    splitStepsBool=False, stepSplit=0.88, timeSplit=0.15,   # optional log-then-linear steps
    # Initial state
    initialTemperature=0.1, radiationTemperature=0.1, colorTemperature=0.1,
    # Boundaries: Reflective, Vacuum, Planckian, Bimodal, Delta, ...
    boundaryLeft="Reflective", boundaryRight="Delta",
    setLeftBoundaryTemp=0.8, setRightBoundaryTemp=0.8,
    deltaFraction=0.5,                # Bimodal: fraction of the energy in the delta spike
    deltaFrequency=None,              # Bimodal: spike photon energy (None = centre of SM group freqNum//4)
    # Solver
    tol=1e-14,
    extrapolateTemp=True,
    andersonDepth=5,                  # 0 = plain fixed-point iteration
    useNumba=True,                    # False = numpy sweeps (identical results, slower)
    opacityModel="nickel",            # "nickel" (TOPS table) or "test" (1/(2T))
    # Manufactured solution (MMSCompare=True adds the MMS sources in both methods)
    MMSCompare=False,
    mmsT0=1.0, mmsSigmaX=1.0, mmsTau=2.0,
    mmsTemperatureOffset=MMS.DEFAULT_TEMPERATURE_OFFSET,
)

SWEEP = dict(              # every combination is run
    nSteps=[2000],
    freqNum=[10, 20, 40, 60, 80, 100, 150],
    tol=[1e-16],
    nBins=[100],
)

# Short names used in run names (others use the parameter name)
SHORT_NAMES = {"freqNum": "G", "nSteps": "Nt", "nBins": "Nx", "sn": "S", "timeMax": "tmax"}

# Parameters that are not in Multigroup.Parameters but are read by the solver
EXTRA_KEYS = {"deltaFraction", "deltaFrequency", "MMSCompare", "mmsT0", "mmsSigmaX", "mmsTau",
              "mmsTemperatureOffset"}


# --- Building and running ------------------------------------------------------------------------
def buildRunSet(base=BASE, sweep=SWEEP):
    """{runName: config} for every combination of the sweep values."""
    keys = list(sweep)
    runs = {}
    for values in itertools.product(*(sweep[k] for k in keys)):
        config = dict(base, **dict(zip(keys, values)))
        name = "_".join(f"{SHORT_NAMES.get(k, k)}{v:g}" if isinstance(v, float) else f"{SHORT_NAMES.get(k, k)}{v}"
                        for k, v in zip(keys, values))
        runs[name or "run"] = config
    return runs


def setRun(params, runName, config, runSet=RUN_SET):
    """Copy a config onto Parameters; unknown keys raise so a typo cannot be silently ignored."""
    unknown = [k for k in config if not hasattr(params, k) and k not in EXTRA_KEYS]
    if unknown:
        raise KeyError(f"Unknown run parameters {unknown} in run {runName!r}")
    for key, value in config.items():
        setattr(params, key, value)
    params.runName = runName
    params.runSet = runSet
    params.saveResults = True
    return params


def runSingle(method, runName, config, runSet=RUN_SET):
    """One solve in a worker process; returns the run's id."""
    constants = Base.Constants()
    params = setRun(MG.Parameters(method), runName, config, runSet)
    grid = Base.Grid(params, constants)
    solver = Base.Base(grid, MG.Problem(grid, constants, params), params, constants)
    solver.solve()
    return f"{method} {runName}"


METHOD_CHOICES = {"SM": ["SM"], "VMG": ["VMG"], "BOTH": ["SM", "VMG"]}


def chooseMethods(argv):
    """Methods from --sm / --vmg / --both (or a bare sm / vmg / both); asks if none is given."""
    parser = argparse.ArgumentParser(description="Run the parameter sweep with SM and/or VMG.")
    group = parser.add_mutually_exclusive_group()
    for name in METHOD_CHOICES:
        group.add_argument(f"--{name.lower()}", dest="choice", action="store_const", const=name,
                           help=f"run {'SM and VMG' if name == 'BOTH' else name}")
    parser.add_argument("method", nargs="?", type=str.upper, choices=list(METHOD_CHOICES),
                        help="same as the flags: sm, vmg or both")
    args = parser.parse_args(argv[1:])
    if args.choice and args.method and args.choice != args.method:
        parser.error(f"--{args.choice.lower()} conflicts with {args.method.lower()}")
    choice = args.choice or args.method
    while choice not in METHOD_CHOICES:
        choice = input("Which method(s)? (SM, VMG, Both): ").strip().upper()
    return METHOD_CHOICES[choice]


def executeRunSet(runs, methods, runSet=RUN_SET):
    """Run every (method, run) pair across the CPU cores; returns the ids of failed runs."""
    tasks = [(method, name, config) for method in methods for name, config in runs.items()]
    print(f"Run set {runSet!r}: {len(tasks)} solves ({', '.join(methods)} x {len(runs)} runs)")
    failed = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(runSingle, method, name, config, runSet): f"{method} {name}"
                   for method, name, config in tasks}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            runId = futures[future]
            try:
                future.result()
                print(f"[{done}/{len(tasks)}] Completed: {runId}")
            except Exception as exc:
                failed.append(runId)
                print(f"[{done}/{len(tasks)}] FAILED: {runId}: {exc}")
    return failed


# Entry point protection required for multiprocessing
if __name__ == "__main__":
    methods = chooseMethods(sys.argv)
    tick = time.perf_counter()
    failed = executeRunSet(buildRunSet(), methods)
    print(f"Set of solves completed in {time.perf_counter() - tick:.2f} seconds.")
    if failed:
        print(f"{len(failed)} failed: " + ", ".join(failed))
