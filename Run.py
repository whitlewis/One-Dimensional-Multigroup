import Marshak
from Marshak import Marshak as MarProblem
import InfiniteVariable
from InfiniteVariable import InfiniteVariable as IVM
import InfiniteMedium
import Reeds
from Reeds import Reeds as ReedsProblem
import Base as Base
import Logic as Log
import MMS
import numpy as np
import Visualize as Vis


class MMSEquations(Log.CoupledEquations):
    def __init__(self, params, grid, material, constants):
        super().__init__(params, grid, material, constants)
        self.T0 = params.mmsT0
        self.sigmaX = params.mmsSigmaX
        self.tau = params.mmsTau
        self.manufacturedQ = np.zeros(params.nBins)

    def planck(self, nu, temperature):
        return MMS.planck(nu, temperature)

    def spatialProfile(self, x):
        return np.exp(-(x**2) / (2 * self.sigmaX**2)) / (
            self.sigmaX * np.sqrt(2 * np.pi)
        )

    def materialTemperature(self, time):
        decay = np.exp(-self.tau * time / 2)
        return self.T0 * (
            (1 - decay) * self.spatialProfile(self.grid.spaceMid)
        ) ** 0.25

    def exactIntensity(self, time, x):
        x = np.asarray(x)
        basePlanck = self.planckBar(
            np.full(x.shape, self.T0)
        )
        decay = np.exp(-self.tau * time / 2)
        return (
            self.const.c / (4 * np.pi)
            * basePlanck
            * (1 + decay)
            * self.spatialProfile(x)[None, :]
        )

    def initialCondition(self):
        intensity = self.exactIntensity(0.0, self.grid.spaceMid)
        return np.broadcast_to(
            intensity[:, None, :],
            (self.params.freqNum, self.sn, self.params.nBins),
        ).copy()

    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        self.grid.temperatureSet[:, 0] = self.materialTemperature(0.0)
        self.grid.T_next = self.materialTemperature(self.grid.timeSet[1]).copy()
        self.T_next = self.grid.T_next.copy()
        self.fullTens = self.grid.fullTensor.copy()
        self.psi_old = self.grid.fullTensor.copy()
        self.grid.fullTensOld = self.grid.fullTensor.copy()
        self.grid.fullTensorPhi = self.getPhi()

    def boundaryCondition(self, side, time):
        x = self.grid.spaceGrid[0] if side == "left" else self.grid.spaceGrid[-1]
        intensity = self.exactIntensity(time, np.array([x]))[:, 0]
        return np.broadcast_to(intensity[:, None], (self.freq, self.sn)).copy()

    def manufacturedSources(self):
        time = self.grid.timeSet[self.grid.timeStep + 1]
        oldTime = self.grid.timeSet[self.grid.timeStep]
        targetTemperature = self.materialTemperature(time)
        oldTemperature = self.materialTemperature(oldTime)
        baseTemperature = np.full(self.params.nBins, self.T0)
        sigmaG = self.material.sigma_a(
            self.grid.freqGroups, targetTemperature
        )
        return MMS.MMSSource1D(
            self.T0,
            baseTemperature,
            baseTemperature,
            time,
            self.grid.dt[self.grid.timeStep],
            self.grid.spaceMid,
            self.mu,
            self.grid.w,
            sigmaG,
            self.material.C_v(targetTemperature),
            self.grid.freqGrid,
            self.sigmaX,
            tau=self.tau,
            materialT=targetTemperature,
            materialTPrevious=oldTemperature,
        )

    def rhsUpdate(self):
        self.startTimeStep()
        temperature = self.grid.T_next
        bbar = self.planckBar(temperature)
        bbar = np.broadcast_to(
            bbar[:, None, :],
            (self.params.freqNum, self.sn, self.params.nBins),
        )
        sigmaG = self.material.sigma_a(self.grid.freqGroups, temperature)
        sigmaByAngle = np.broadcast_to(
            sigmaG[:, None, :],
            (self.params.freqNum, self.sn, self.params.nBins),
        )
        sourceS, self.manufacturedQ = self.manufacturedSources()
        self.grid.rhs = (
            4 * np.pi * sigmaByAngle * bbar
            + self.timeTerm * self.psi_old
            + sourceS
        )
        self.sigmaStarVar = self.sigmaStar(temperature)

    def materialEquation(self):
        dt = self.grid.dt[self.grid.timeStep]
        temperatureOld = self.grid.temperatureSet[:, self.grid.timeStep]
        temperature = self.grid.T_next
        heatCapacity = self.material.C_v(temperatureOld)
        factor = dt / heatCapacity
        phi = self.getPhi()
        bbar = self.planckBar(temperature)
        dbbar = self.planckBarDeriv(temperature)
        sigmaG = self.material.sigma_a(self.grid.freqGroups, temperature)

        residual = temperatureOld - temperature + factor * (
            np.sum(sigmaG * phi - 4 * np.pi * sigmaG * bbar, axis=0)
            + self.manufacturedQ
        )
        correction = residual / (
            1 + factor * 4 * np.pi * np.sum(sigmaG * dbbar, axis=0)
        )
        correction = np.maximum(correction, -0.5 * temperature)
        temperatureNext = np.maximum(
            temperature + correction, 1e-12
        )
        self.T_next = temperatureNext.copy()
        self.grid.T_next = temperatureNext.copy()


class MMSProblem:
    def __init__(self, grid, constants, params):
        self.material = MMSMaterial(params, grid)
        self.equations = Log.Logic(params, grid, self.material, constants)
        self.equations.equations = MMSEquations(
            params, grid, self.material, constants
        )
        self.equations.materialEquation = (
            self.equations.equations.materialEquation
        )
        self.equations.applyInitialConditions()


class MMSMaterial(InfiniteMedium.Material):
    def sigma_a(self, freq, temperature):
        return np.full(
            (self.params.freqNum, self.params.nBins),
            self.params.mmsOpacity,
        )

    def C_v(self, temperature):
        return self.params.mmsHeatCapacity


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
    InfiniteVariableProblem = IVM(grid, constants, params)
    BaseSolver = Base.Base(grid, InfiniteVariableProblem, params,  constants)
    fullTensor, grid = BaseSolver.solve()
    fullPhi = np.squeeze(grid.fullTensorPhiTime)[:-1,:,:]
    Vis.plotTemperatureTime(grid)
    Vis.plotTemperature(grid)

    Vis.plot_spectra_at_times(fullPhi, plotSet, 20, params.maxFreq, freqs=grid.freqGroups)
    rankInfo = Vis.analyzeRank(fullPhi, BaseSolver.grid.timeSet, time_indices=None, energy_threshold=[0.999, 0.99, 0.98  ,0.95], tol=None, plot=True)
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


def mainMMS(nSteps=100, nBins=100, opacity=0.1):
    """Run the coupled Gaussian MMS solve and plot its scalar flux and temperature."""
    constants = Base.Constants()
    params = InfiniteMedium.Parameters(nSteps=nSteps)
    params.fileFolder = "MMS"
    params.runName = "Gaussian1D_MMS"
    params.saveResults = False
    params.xMin = -2.0
    params.xMax = 2.0
    params.nBins = nBins
    params.initialTemperature = 0.0
    params.boundaryLeft = "Planckian"
    params.boundaryRight = "Planckian"
    params.freqNum = 24
    params.minFreq = 1e-3
    params.maxFreq = 3.0
    params.infFreq = 5.0
    params.groupSpace = "log"
    params.matchNu = False
    params.nSteps = nSteps
    params.timeMax = 1.0
    params.timeScale = "linear"
    params.mmsT0 = 0.25
    params.mmsSigmaX = 0.5
    params.mmsTau = 2.0
    params.mmsOpacity = opacity
    params.mmsHeatCapacity = 0.01

    grid = Base.Grid(params, constants)
    problem = MMSProblem(grid, constants, params)
    solver = Base.Base(grid, problem, params, constants)
    _, grid = solver.solve()
    Vis.plotTemperatureTime(grid)
    fullPhi = np.sum(grid.fullTensorPhiTime, axis=1)
    Vis.plotSpaceTime(fullPhi, params, grid.timeSet)
    return grid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the existing moving-mesh solve or the Gaussian MMS solve."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("movingmesh", "no-mms", "mms"),
        default="movingmesh",
        help=(
            "'movingmesh' and 'no-mms' use the existing solver and initial "
            "conditions; 'mms' enables the manufactured sources."
        ),
    )
    runMode = parser.parse_args().mode
    if runMode == "mms":
        grid = mainMMS()
    else:
        grid = mainMovingMesh()
