"""Single-run driver: one SM or VMG solve with plots, or the constant-opacity Gaussian MMS solve.

    python driver/Run.py          # VMG (default)
    python driver/Run.py sm       # SM
    python driver/Run.py mms      # SM with manufactured sources and constant opacity
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("src", "postprocessing"):
    if os.path.join(_ROOT, _folder) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _folder))

import numpy as np

import AnalyzeData
import Base
import Logic as Log
import MMS
import Multigroup as MG
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
        profile = (1 - decay) * self.spatialProfile(self.grid.spaceMid)
        return MMS.temperatureFromFourthPowerProfile(
            self.T0,
            profile,
            getattr(
                self.params,
                "mmsTemperatureOffset",
                MMS.DEFAULT_TEMPERATURE_OFFSET,
            ),
        )

    def exactIntensity(self, time, x):
        x = np.asarray(x)
        basePlanck = self.planckBar(
            np.full(x.shape, self.T0)
        )
        offsetPlanck = self.planckBar(
            np.full(
                x.shape,
                getattr(
                    self.params,
                    "mmsTemperatureOffset",
                    MMS.DEFAULT_TEMPERATURE_OFFSET,
                ),
            )
        )
        decay = np.exp(-self.tau * time / 2)
        return (
            basePlanck * (1 + decay)
            * self.spatialProfile(x)[None, :]
            + offsetPlanck
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
            temperatureOffset=getattr(
                self.params,
                "mmsTemperatureOffset",
                MMS.DEFAULT_TEMPERATURE_OFFSET,
            ),
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


class MMSMaterial(MG.Material):
    def sigma_a(self, freq, temperature):
        return np.full(
            (self.params.freqNum, self.params.nBins),
            self.params.mmsOpacity,
        )

    def C_v(self, temperature):
        return self.params.mmsHeatCapacity


def mainSingle(method):
    """Run one SM or VMG solve with the default Parameters, then plot it like a loaded result."""
    constants = Base.Constants()
    params = MG.Parameters(method)
    params.storeAngular = True  # analyze_rank_psi needs the full angular history
    grid = Base.Grid(params, constants)
    problem = MG.Problem(grid, constants, params)
    solver = Base.Base(grid, problem, params, constants)
    fullTensor, grid = solver.solve()

    Vis.plotTemperatureTime(grid)
    Vis.plotTemperature(grid)
    data, runParams, fileName, folder = AnalyzeData.resultFromGrid(grid, params)
    AnalyzeData.plotResults([data], [runParams], [fileName], [folder])
    rankInfo = Vis.analyzeRank([data], grid.timeSet, [runParams], [fileName], [folder],
                               energy_threshold=[0.999, 0.99, 0.98, 0.95], plot=True)
    Vis.analyzeRank([data], grid.timeSet, [runParams], [fileName], [folder],
                    tol=[1e-8, 1e-14, 1e-16, 1e-18], plot=True)
    Vis.analyze_rank_psi(fullTensor, grid.timeSet, plot=True)
    return grid, rankInfo


def mainSM():
    return mainSingle("SM")


def mainVMG():
    return mainSingle("VMG")


def mainMMS(nSteps=100, nBins=100, opacity=0.1):
    """Run the coupled Gaussian MMS solve and plot its scalar flux and temperature."""
    constants = Base.Constants()
    params = MG.Parameters("SM", nSteps=nSteps)
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
    params.mmsTemperatureOffset = MMS.DEFAULT_TEMPERATURE_OFFSET
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
        description="Run one SM or VMG solve, or the Gaussian MMS solve."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("vmg", "sm", "mms"),
        default="vmg",
        help="'vmg' or 'sm' run that method with its default parameters; 'mms' runs the Gaussian MMS solve.",
    )
    runMode = parser.parse_args().mode
    if runMode == "mms":
        grid = mainMMS()
    elif runMode == "sm":
        grid, rankInfo = mainSM()
    else:
        grid, rankInfo = mainVMG()
