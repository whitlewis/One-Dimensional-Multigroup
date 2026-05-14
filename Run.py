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
    sol = BaseSolver.solve()
    solPlot = np.squeeze(sol)
    Vis.plotSpaceTime(solPlot[:,0,0,:], params, grid.dt)
    Vis.animate_solution(solPlot[:,0,0,:], params, grid)

mainReeds()
