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
    params.nSteps = config["Steps"]
    params.timeMax = config["Maximum Time"]
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
    },
    "VacuumShort": {
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
    },
    "PlanckShort": {
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
    },
        "DeltaShort": {
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
    },
    "Reflective": {
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
    },
        "Vacuum": {
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
    },
    "Planck": {
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
    },
        "Delta": {
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
    },
    # "ReflectiveLong": {
    #     "Steps": 5000,
    #     "Maximum Time": 2.0,
    #     "Max Frequency": 20,
    #     "Number of frequencies": 100,
    #     "Sn": 8,
    #     "Number of Bins": 100,
    #     "xMin": -1,
    #     "xMax": 1,
    #     "RadTemp": 0.5,
    #     "MatTemp": 0.4,
    #     "leftBC": "Reflective",
    #     "rightBC": "Reflective",
    #     "leftTemp": 0.5,
    #     "rightTemp": 0.5,
    # },
    #     "VacuumLong": {
    #     "Steps": 5000,
    #     "Maximum Time": 2.0,
    #     "Max Frequency": 20,
    #     "Number of frequencies": 100,
    #     "Sn": 8,
    #     "Number of Bins": 100,
    #     "xMin": -1,
    #     "xMax": 1,
    #     "RadTemp": 0.5,
    #     "MatTemp": 0.4,
    #     "leftBC": "Vacuum",
    #     "rightBC": "Vacuum",
    #     "leftTemp": 0.5,
    #     "rightTemp": 0.5,
    # },
    # "PlanckLong": {
    #     "Steps": 5000,
    #     "Maximum Time": 2.0,
    #     "Max Frequency": 20,
    #     "Number of frequencies": 100,
    #     "Sn": 8,
    #     "Number of Bins": 100,
    #     "xMin": -1,
    #     "xMax": 1,
    #     "RadTemp": 0.5,
    #     "MatTemp": 0.4,
    #     "leftBC": "Planckian",
    #     "rightBC": "Planckian",
    #     "leftTemp": 0.5,
    #     "rightTemp": 0.5,
    # },
    #     "DeltaLong": {
    #     "Steps": 5000,
    #     "Maximum Time": 2.0,
    #     "Max Frequency": 20,
    #     "Number of frequencies": 100,
    #     "Sn": 8,
    #     "Number of Bins": 100,
    #     "xMin": -1,
    #     "xMax": 1,
    #     "RadTemp": 0.5,
    #     "MatTemp": 0.4,
    #     "leftBC": "Delta",
    #     "rightBC": "Delta",
    #     "leftTemp": 0.5,
    #     "rightTemp": 0.5,
    # },


}

# Entry point protection required for multiprocessing
if __name__ == "__main__":
    tick = time.perf_counter()
    completed_results = ExecuteRunSetParallel(RunSet)
    tock = time.perf_counter()
    print(f"Set of solves completed in {tock - tick:.2f} seconds.")    