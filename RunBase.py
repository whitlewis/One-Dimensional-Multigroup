import Marshak
from Marshak import Marshak as MarProblem
import InfiniteMedium   
from InfiniteMedium import InfiniteMedium as IM
import Reeds
from Reeds import Reeds as ReedsProblem
import Base as Base
import numpy as np
import Visualize as Vis


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
    nSteps = params.nSteps
    plotSet = [0, nSteps//4, nSteps//2, nSteps-1]
    Vis.plot_spectra_at_times(fullPhi, plotSet, 20, params, freqs=grid.freqGroups)
    rankInfo = Vis.analyze_rank(time_indices=plotSet, energy_threshold=0.99)
    return grid, rankInfo, plotSet


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

grid, rankInfo, plotSet = mainInfinite()
print("Rank information at specified time steps:")
rank, singular_values = rankInfo
print(f"Ranks at time steps {plotSet}: {rank}")
