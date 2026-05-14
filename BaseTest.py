import numpy as np
import numpy.polynomial.legendre as leggauss

class Constants:
    c = 1.0
    a = 0.01374
    h = 1.0


class Grid:
    def __init__(self, parameters, Constants=Constants()):

        self.dx = (
            parameters.xMax - parameters.xMin
        ) / parameters.nBins

        self.spaceGrid = np.linspace(
            parameters.xMin,
            parameters.xMax,
            parameters.nBins + 1,
        )

        self.freqGrid = np.append(
            np.logspace(
                -12,
                np.log10(25),
                parameters.freqNum - 1,
            ),
            parameters.maxFreq,
        )

        self.fullTensor = np.zeros(
            (
                parameters.freqNum,
                parameters.sn,
                parameters.nBins,
            )
        )

        self.muSet, self.w = leggauss.leggauss(parameters.sn)

        self.w /= 2.0

        self.timeSet = np.logspace(
            -12,
            np.log10(parameters.timeMax),
            parameters.nSteps,
        )

        self.dt = np.diff(self.timeSet)

        self.timeStep = 0

        self.temperatureSet = np.zeros(
            (parameters.nBins, parameters.times)
        )

        self.fullTensorTime = np.zeros(
            (parameters.times,) + self.fullTensor.shape
        )

        self.fullTensorPhi = np.zeros(
            (parameters.freqNum, parameters.nBins)
        )

        self.fullTensorPhiTime = np.zeros(
            (parameters.times,) + self.fullTensorPhi.shape
        )


class Base:
    def __init__(self, grid, problem, constants=Constants()):

        self.grid = grid
        self.problem = problem
        self.constants = constants

        self.params = problem.parameters

        self.fullTensor = grid.fullTensor.copy()

    def converge(self):

        self.fullTensorOld = self.grid.fullTensor.copy() * 0

        for it in range(self.params.maxIters):

            newFull = self.problem.equations.radiationSweep(
                self.grid.temperatureSet[:, self.grid.timeStep]
            )

            err = np.max(
                np.abs(newFull - self.fullTensorOld)
            )

            if err < self.params.tol:

                print(f"Converged in {it} iterations")

                break

            self.fullTensorOld = newFull.copy()

        return newFull

    def getPhi(self, fullTensor):

        fullTensorPhi = np.sum(
            self.grid.w[None, :, None] * fullTensor,
            axis=1,
        )

        return fullTensorPhi

    def solve(self):

        for index, time in enumerate(self.grid.timeSet[:-1]):

            newFull = self.converge()

            self.fullTensor = newFull.copy()

            self.grid.fullTensorTime[index] = newFull.copy()

            self.fullTensorPhi = self.getPhi(newFull)

            self.grid.fullTensorPhiTime[index] = (
                self.fullTensorPhi.copy()
            )

            self.grid.timeStep += 1

        return self.grid.fullTensorTime