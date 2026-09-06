import concurrent.futures
import time

# Standard Imports
import InfiniteMedium   
from InfiniteMedium import InfiniteMedium as IM
import Reeds
from Reeds import Reeds as ReedsProblem
import Base as Base

# Variable Coordinate Imports
import InfiniteVariable
from InfiniteVariable import InfiniteVariable as IVM
import Reeds
from Reeds import Reeds as ReedsProblem
import Base as Base
import numpy as np


def setRun(params, runName, config):
    params.runName = runName
    params.runLabel = config["runLabel"]
    params.nSteps = config["Steps"]
    params.timeMax = config["Maximum Time"]
    params.logLinTime = config["TransTime"]
    params.stepSplit = config["stepSplit"]  # tells what proportion of time steps are log vs linear
    params.splitStepsBool = config["stepType"]  # tells whether to split time stepping
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

    # Choose to save
    params.saveResults = True
    return params



# Single-run worker functions for parallel execution
def run_single_standard(item):
    runName, config = item
    constants = Base.Constants()
    params = InfiniteMedium.Parameters()
    setRun(params, runName, config)

    grid = Base.Grid(params, constants)
    problem = IM(grid, constants, params)
    solver = Base.Base(grid, problem, params, constants)

    fullTensor, grid = solver.solve()
    return f"{runName}_IM", fullTensor, grid


def run_single_variable(item):
    runName, config = item
    constants = Base.Constants()
    params = InfiniteVariable.Parameters()
    params = setRun(params, runName, config)

    grid = Base.Grid(params, constants)
    problem = IVM(grid, constants, params)
    solver = Base.Base(grid, problem, params, constants)
    fullTensor, grid = solver.solve()
    return f"{runName}_IVM", fullTensor, grid


def ExecuteRunSetParallel(runSet):
    methodChoice = input("Which method is desired? (IVM, IM, Both): ").strip().lower()

    tasks = []
    if methodChoice in ("im", "both"):
        tasks.extend([(run_single_standard, item) for item in runSet.items()])
    if methodChoice in ("ivm", "both"):
        tasks.extend([(run_single_variable, item) for item in runSet.items()])

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
                run_id, fullTensor, grid = future.result()
                results[run_id] = {"tensor": fullTensor, "grid": grid}
                print(f"Completed: {run_id}")
            except Exception as exc:
                print(f"Task '{task_name}' generated an exception: {exc}")

    return results


RunSet = {
    "ReflectiveShort": {
        "runLabel": "Reflective",
        "Steps": 400,
        "Maximum Time": 0.1,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Reflective",
        "rightBC": "Reflective",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
    "VacuumShort": {
        "runLabel": "Vacuum",
        "Steps": 400,
        "Maximum Time": 0.1,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Vacuum",
        "rightBC": "Vacuum",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
    "PlanckShort": {
        "runLabel" : "Planckian",
        "Steps": 400,
        "Maximum Time": 0.1,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Planckian",
        "rightBC": "Planckian",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
        "DeltaShort": {
        "runLabel" : "Delta",
        "Steps": 400,
        "Maximum Time": 0.1,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Delta",
        "rightBC": "Delta",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
    "Reflective": {
        "runLabel" : "Reflective",
        "Steps": 4000,
        "Maximum Time": 1.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Reflective",
        "rightBC": "Reflective",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
        "Vacuum": {
        "runLabel" : "Vacuum",
        "Steps": 4000,
        "Maximum Time": 1.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Vacuum",
        "rightBC": "Vacuum",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
    "Planck": {
        "runLabel" : "Planckian",
        "Steps": 4000,
        "Maximum Time": 1.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Planckian",
        "rightBC": "Planckian",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
        "Delta": {
        "runLabel" : "Delta",
        "Steps": 4000,
        "Maximum Time": 1.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Delta",
        "rightBC": "Delta",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
    "ReflectiveLong": {
        "runLabel" : "Reflective",
        "Steps": 8000,
        "Maximum Time": 2.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Reflective",
        "rightBC": "Reflective",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
        "VacuumLong": {
        "runLabel" : "Vacuum",
        "Steps": 8000,
        "Maximum Time": 2.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Vacuum",
        "rightBC": "Vacuum",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
    "PlanckLong": {
        "runLabel" : "Planckian",
        "Steps": 8000,
        "Maximum Time": 2.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Planckian",
        "rightBC": "Planckian",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },
        "DeltaLong": {
        "runLabel" : "Delta",
        "Steps": 8000,
        "Maximum Time": 2.0,
        "Max Frequency": 20,
        "Number of frequencies": 100,
        "Sn": 8,
        "Number of Bins": 100,
        "xMin": -1,
        "xMax": 1,
        "RadTemp": 0.5,
        "MatTemp": 0.4,
        "leftBC": "Delta",
        "rightBC": "Delta",
        "leftTemp": 0.5,
        "rightTemp": 0.5,
        "TransTime": 0.2,
        "stepSplit": 0.25,  # tells what proportion of time steps are log vs linear
        "stepType": True, 
    },


}

# Entry point protection required for multiprocessing
if __name__ == "__main__":
    tick = time.perf_counter()
    completed_results = ExecuteRunSetParallel(RunSet)
    tock = time.perf_counter()
    print(f"Set of solves completed in {tock - tick:.2f} seconds.")    