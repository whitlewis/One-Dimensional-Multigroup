import Marshak
from Marshak import Marshak as MarProblem
from InfiniteMedium import InfiniteMedium
import Reeds
from Reeds import Reeds as ReedsProblem
import Base as Base
import numpy as np
import Visualize as Vis

def main():
    constants = Base.Constants()
    params = Marshak.Parameters()
    grid = Base.Grid(params, constants)
    MarshakProblem = MarProblem(grid, constants)
    BaseSolver = Base.Base(grid, MarshakProblem, params,  constants)
    print(np.shape(grid.fullTensor))
    print(f'Initial condition check: {grid.fullTensor[:, :, 0]}')  # Print the initial condition for the first spatial bin

    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)  # shape: (nSteps+1, nBins)
    Vis.plotSpaceTime(fullPhi, params, grid.timeSet)
    Vis.plotFinalFlux(grid)
    # Vis.animate_solution(fullPhi, params, grid)
    Vis.plotTemperatureTime(grid)
    return grid

def mainInfinite():
    constants = Base.Constants()
    params = InfiniteMedium.Parameters()
    grid = Base.Grid(params, constants)
    InfiniteMediumProblem = InfiniteMedium.InfiniteMedium(grid, constants)
    BaseSolver = Base.Base(grid, InfiniteMediumProblem, params,  constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)  # shape: (nSteps+1, nBins)
    Vis.plotSpaceTime(fullPhi, params, grid.timeSet)
    Vis.plotFinalFlux(grid)
    Vis.animate_solution(fullPhi, params, grid)
    return grid


def mainReeds():
    constants = Base.Constants()
    params = Reeds.Parameters()
    grid = Base.Grid(params, constants)
    ReedsProb = ReedsProblem(grid, constants)
    BaseSolver = Base.Base(grid, ReedsProb, params, constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)  # shape: (nSteps+1, nBins)
    Vis.plotSpaceTime(fullPhi, params, grid.timeSet)
    Vis.plotFinalFlux(grid)
    Vis.animate_solution(fullPhi, params, grid)
    return grid

grid = main()
