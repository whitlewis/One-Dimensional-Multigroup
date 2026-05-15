import Marshak
from Marshak import Marshak as MarProblem
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
    MarshakProblem.equations.initSpectra()
    BaseSolver = Base.Base(grid, MarshakProblem, params,  constants)
    sol = BaseSolver.solve()
    Vis.PlotTemperature(grid)


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

grid = mainReeds()
