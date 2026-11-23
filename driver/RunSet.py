"""Batch driver: run a set of SM and/or VMG solves in parallel and save each to dataStash/."""
import concurrent.futures
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


def setRun(params, runName, config):
    params.runName = runName
    params.tol = config.get("Iteration Tolerance", params.tol)
    params.runLabel = config["runLabel"]
    params.groupSpace = config["Group Spacing"] # log or linear
    params.matchNu = config["Match Grid"]
    params.minFreq = config["Min Frequency"]
    params.infFreq = config["Infinite Frequency"]
    params.nSteps = config["Steps"]
    params.timeMax = config["Maximum Time"]
    params.timeSplit = config["TransTime"]
    params.stepSplit = config["stepSplit"]  # tells what proportion of time steps are log vs linear
    params.splitStepsBool = config["stepType"]  # tells whether to split time stepping
    params.timeScale = config["timeScale"]
    params.maxFreq = config["Max Frequency"]
    params.freqNum = config["Number of frequencies"]
    params.sn = config["Sn"]
    params.boundaryLeft = config["leftBC"]
    params.boundaryRight = config["rightBC"]
    params.setLeftBoundaryTemp = config["leftTemp"]
    params.setRightBoundaryTemp = config["rightTemp"]
    params.nBins = config["Number of Bins"]
    params.xMin = config["xMin"]
    params.xMax = config["xMax"]
    params.initialTemperature = config["MatTemp"]
    params.radiationTemperature = config["RadTemp"]
    params.colorTemperature = config["ColorTemp"]

    # Debug params
    params.extrapolateTemp = config.get("extrapolateTemp", False)
    params.andersonDepth = config.get("Anderson Depth", getattr(params, "andersonDepth", 0))
    if not config.get("Use Anderson", True):  # "Use Anderson": False turns acceleration off (plain iteration)
        params.andersonDepth = 0
    params.useNumba = config.get("Use Numba", getattr(params, "useNumba", True))
    # Bimodal boundary: fraction of the boundary Planckian's energy put in the delta spike, and the
    # spike's physical photon energy (None = centre of SM group freqNum//4)
    params.deltaFraction = config.get("Delta Fraction", getattr(params, "deltaFraction", 0.5))
    params.deltaFrequency = config.get("Delta Frequency", getattr(params, "deltaFrequency", None))
    params.opacityModel = config.get("Opacity Model", getattr(params, "opacityModel", "nickel"))
    params.MMSCompare = config.get("MMSCompare", False)
    params.mmsT0 = config.get("MMS T0", params.radiationTemperature)
    params.mmsSigmaX = config.get("MMS Sigma X", 0.5)
    params.mmsTau = config.get("MMS Tau", 2.0)
    params.mmsTemperatureOffset = config.get(
        "MMS Temperature Offset", MMS.DEFAULT_TEMPERATURE_OFFSET
    )


    # Choose to save
    params.saveResults = True
    return params



# Single-run worker functions for parallel execution
def run_single(method, item):
    runName, config = item
    constants = Base.Constants()
    params = setRun(MG.Parameters(method), runName, config)
    grid = Base.Grid(params, constants)
    problem = MG.Problem(grid, constants, params)
    solver = Base.Base(grid, problem, params, constants)
    solver.solve()
    return f"{runName}_{params.fileFolder}"


def run_single_sm(item):
    return run_single("SM", item)


def run_single_vmg(item):
    return run_single("VMG", item)


def ExecuteRunSetParallel(runSet):
    methodChoice = input("Which method is desired? (SM, VMG, Both): ").strip().lower()

    tasks = []
    if methodChoice in ("sm", "both"):
        tasks.extend([(run_single_sm, item) for item in runSet.items()])
    if methodChoice in ("vmg", "both"):
        tasks.extend([(run_single_vmg, item) for item in runSet.items()])
    if not tasks:
        print(f"Unknown method {methodChoice!r}: choose SM, VMG, or Both.")

    results = {}

    # Distribute tasks across CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks: worker_func(item)
        future_to_task = {
            executor.submit(worker, item): item[0] for worker, item in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                run_id = future.result()
                print(f"Completed: {run_id}")
            except Exception as exc:
                print(f"Task '{task_name}' generated an exception: {exc}")

    return results



stepSplit = 0.88
timeSplit = 0.15

# groupSet = [10, 50, 100, 150]
# stepSet = [2500, 5000]
groupSet = [100]
stepSet = [500, 1000, 1500, 2000, 3000, 4000, 8000]
tolSet = [1e-14]

RunSet = {}
s = 0
g = 0


for step in stepSet:
    s += 1
    for group in groupSet:
        g += 1
        for tol in tolSet:
            runName = f"BimodalBC_{s}_{g}_{tol}"
            RunSet[runName] = {
                "runLabel": "Planck",
                "Steps": step,
                "Iteration Tolerance": tol,
                "Maximum Time": 0.5,
                "Max Frequency": 45,
                "Min Frequency": 1e-4,
                "Infinite Frequency": 125,
                "Group Spacing": 'log',
                "Match Grid" : True,
                "Number of frequencies": group,
                "Sn": 8,
                "Number of Bins": 50,
                "xMin": -1,
                "xMax": 1,
                "RadTemp": 0.1,
                "MatTemp": 0.1,
                "ColorTemp" : 0.1,
                "leftBC": "Reflective",
                "rightBC": "Bimodal",
                "leftTemp": 0.8,
                "rightTemp": 0.8,
                "timeScale": "log",
                "TransTime": timeSplit,
                "stepSplit": stepSplit,  # tells what proportion of time steps are log vs linear
                "stepType": False, 
                "extrapolateTemp": True,
                "Use Anderson": True,  # False = plain fixed-point iteration
                "Anderson Depth": 5,   # history length when Anderson is on
                "Use Numba": True,     # False = numpy/Python sweeps (identical results, slower)
                "MMSCompare": False,  # Set True to add MMS S and Q in both methods.
                "MMS T0": 1.0,
                "MMS Sigma X": 1.0,
                "MMS Tau": 2.0,
                "MMS Temperature Offset": MMS.DEFAULT_TEMPERATURE_OFFSET,
            }
        # runName = f"PlanckLinear{s}_{g}"
        # RunSet[runName] = {
        #     "runLabel": "MMSLinear",
        #     "Steps": step,
        #     "Iteration Tolerance": 1e-20,
        #     "Maximum Time": 0.5,
        #     "Max Frequency": 45,
        #     "Min Frequency": 1e-4,
        #     "Infinite Frequency": 125,
        #     "Group Spacing": 'log',
        #     "Match Grid" : True,
        #     "Number of frequencies": group,
        #     "Sn": 10,
        #     "Number of Bins": 100,
        #     "xMin": -1,
        #     "xMax": 1,
        #     "RadTemp": 0.5,
        #     "MatTemp": 0.5,
        #     "ColorTemp" : 0.5,
        #     "leftBC": "Reflective",
        #     "rightBC": "Planckian",
        #     "leftTemp": 0.8,
        #     "rightTemp": 0.8,
        #     "timeScale": "linear",
        #     "TransTime": timeSplit,
        #     "stepSplit": stepSplit,  # tells what proportion of time steps are log vs linear
        #     "stepType": False, 
        #     "extrapolateTemp": True,
        #     "MMSCompare": False,  # Set True to add MMS S and Q in both methods.
        #     "MMS T0": 1.0,
        #     "MMS Sigma X": 1.0,
        #     "MMS Tau": 2.0,
        #     "MMS Temperature Offset": MMS.DEFAULT_TEMPERATURE_OFFSET,
        # }

# Entry point protection required for multiprocessing
if __name__ == "__main__":
    tick = time.perf_counter()
    completed_results = ExecuteRunSetParallel(RunSet)
    tock = time.perf_counter()
    print(f"Set of solves completed in {tock - tick:.2f} seconds.")    