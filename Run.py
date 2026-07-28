import Marshak
from Marshak import Marshak as MarProblem
import InfiniteMedium   
from InfiniteMedium import InfiniteMedium as IM
import InfiniteVariable
from InfiniteVariable import InfiniteVariable as IVM
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
    fullPhi = np.squeeze(grid.fullTensorPhiTime)
    Vis.plotSpaceTime(fullPhi[:,5,:], params, grid.timeSet)
    Vis.plotFinalFlux(grid)
    # Vis.animate_solution(fullPhi, params, grid)
    Vis.plotTemperatureTime(grid)
    return grid

def mainInfinite():
    constants = Base.Constants()
    params = InfiniteMedium.Parameters()
    grid = Base.Grid(params, constants)
    InfiniteMediumProblem = IM(grid, constants) # This can change to IVM for the infinite variable problem
    BaseSolver = Base.Base(grid, InfiniteMediumProblem, params,  constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)
    Vis.plotSpaceTime(fullPhi[:,5,:], params, grid.timeSet)
    Vis.plotFinalFlux(grid)
    # Vis.animate_solution(fullPhi, params, grid)
    Vis.plotTemperatureTime(grid)
    Vis.plotTemperature(grid)
    return grid

def mainMovingMesh():
    constants = Base.Constants()
    params = InfiniteVariable.Parameters()
    grid = Base.Grid(params, constants)
    InfiniteVariableProblem = IVM(grid, constants)
    BaseSolver = Base.Base(grid, InfiniteVariableProblem, params,  constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)
    Vis.plotSpaceTime(fullPhi[:,5,:], params, grid.timeSet)
    Vis.plotFinalFlux(grid)
    # Vis.animate_solution(fullPhi, params, grid)
    Vis.plotTemperatureTime(grid)
    Vis.plotTemperature(grid)
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

grid = mainInfinite()
# grid = mainMovingMesh()
