import Marshak
from Marshak import Marshak as MarProblem
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



def mainMovingMesh():
    constants = Base.Constants()
    params = InfiniteVariable.Parameters()
    nSteps = params.nSteps
    plotSet = [0, nSteps//4, nSteps//2, nSteps-1]
    grid = Base.Grid(params, constants)
    InfiniteVariableProblem = IVM(grid, constants)
    BaseSolver = Base.Base(grid, InfiniteVariableProblem, params,  constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)[:-1,:,:]
    # Vis.plotSpaceTime(fullPhi[:,5,:], params, grid.timeSet)s
    # Vis.plotFinalFlux(grid)
    # Vis.animate_solution(fullPhi, params, grid)
    Vis.plotTemperatureTime(grid)
    Vis.plotTemperature(grid)

    Vis.plot_spectra_at_times(fullPhi, plotSet, 20, params.maxFreq, freqs=grid.freqGroups)
    rankInfo = Vis.analyzeRank(fullPhi, BaseSolver.grid.timeSet, time_indices=None, energy_threshold=[.999999999, .999999, .9999  ,.99], tol=None, plot=True)
    Vis.analyzeRank(fullPhi, BaseSolver.grid.timeSet, energy_threshold=None, tol = [1e-8, 1e-14, 1e-16, 1e-18], plot = True)
    Vis.analyze_rank_psi(fullTensor, BaseSolver.grid.timeSet, time_indices=None, energy_threshold=None, tol=None, plot=True)
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

grid = mainMovingMesh()
