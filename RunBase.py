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
    nSteps = params.nSteps
    plotSet = [0, nSteps//4, nSteps//2, nSteps-1]
    grid = Base.Grid(params, constants)
    InfiniteMediumProblem = IM(grid, constants) # This can change to IVM for the infinite variable problem
    BaseSolver = Base.Base(grid, InfiniteMediumProblem, params,  constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)[:-1,:,:]
    # Vis.plotSpaceTime(fullPhi[:,5,:], params, grid.timeSet)
    # Vis.plotFinalFlux(grid)
    # Vis.animate_solution(fullPhi, params, grid)
    Vis.plotTemperatureTime(grid)
    Vis.plotTemperature(grid)

    Vis.plot_spectra_at_times(fullPhi, plotSet, 20, params, freqs=grid.freqGroups)
    rankInfo = Vis.analyzeRank(fullPhi, BaseSolver.grid.timeSet, time_indices=None, energy_threshold=[.999999999, .999999, .9999  ,.99], tol=None, plot=True)
    Vis.analyzeRank(fullPhi, BaseSolver.grid.timeSet, energy_threshold=None, tol = [1e-8, 1e-14, 1e-16, 1e-18], plot = True)
    # Vis.analyze_rank_psi(fullTensor, BaseSolver.grid.timeSet, time_indices=None, energy_threshold=[.99, .95, .9], tol=1e-16, plot=True)
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

