import numpy as np
import numpy.polynomial.legendre as leggauss

class Parameters:
    def __init__(self, maxIters=100, tol=1e-10, nSteps=1000, Transient=True):
        self.maxIters = maxIters
        self.tol = tol
        self.nSteps = nSteps
        self.transient = Transient
        self.nBins = 120
        self.xMin = -8
        self.xMax = 8
        self.sn = 8
        self.freqNum = 10
        self.timeMax = 20.0
        self.maxFreq = 150
        self.initialTemperature = 0.0
        self.sourceTemp = 1.0
        self.timeScale = "linear"  # "log" or "linear"

class Material:
    def __init__(self, params, grid):
        self.params = params
        self.grid = grid

    def sigma_a(self, freq, T):
        freq = np.asarray(freq)
        T = np.asarray(T)
        if freq.ndim == 1:
            freq = freq[:, None]
        if T.ndim == 1:
            T = T[None, :]
        return 100.0 * np.ones_like(freq * T)

    def C_v(self, T):  # Placeholder constant heat capacity
        T = np.asarray(T)
        return np.ones_like(T)

    def addSource(self):
        source = np.zeros((self.grid.freqNum, self.grid.sn))
        source[:, :] = self.params.sourceTemp
        return source


class Equations:

    def __init__(self, params, grid, material, constants):
        self.params = params
        self.grid = grid
        self.material = material
        self.const = constants
        self.fullTens = grid.fullTensor.copy()  # shape: (freqNum, sn, nBins)
        self.freq = params.freqNum
        self.sn = params.sn
        self.dx = grid.dx
        self.time_step = None

    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
        return out

    def planck(self, nu, T):  # Planck function (not group integrated or weighted)
        denom = np.expm1(nu/T)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * nu**3 / denom

    def groupPlanck(self, T):
        freq = np.asarray(self.grid.freqGrid)
        if freq.ndim == 1 and freq.shape[0] == self.params.freqNum + 1:
            lo = freq[:-1, None]
            hi = freq[1:, None]
            return self.simpson(lambda nu: self.planck(nu, T), lo, hi)

        if freq.ndim == 1:
            freq = freq[:, None]
        T = np.asarray(T)
        if T.ndim == 1:
            T = T[None, :]
        return self.planck(freq, T)

    def initSpectra(self):
        T0 = self.params.initialTemperature
        planck = self.groupPlanck(T0) 
        self.grid.fullTensor[:] = planck[:, None, None]
        self.grid.updateFullTensor(self.grid.fullTensor)

    def initialCondition(self):
        return np.zeros_like(self.grid.fullTensor)

    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        self.fullTens = self.grid.fullTensor.copy()
        self.psi_old = self.grid.fullTensor.copy()

    def boundaryCondition(self, side, time):
        return np.zeros((self.freq, self.sn))

    def timeAbsorption(self):
        if not self.params.transient:
            return 0.0
        return 1.0 / (self.const.c * self.grid.dt[self.grid.timeStep])

    def startTimeStep(self):
        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.fullTens = self.grid.fullTensor.copy()

    def materialEquation(self, fullTensor):
        self.startTimeStep()
        dt = self.grid.dt[self.grid.timeStep]
        f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep])
        T = self.grid.temperatureSet[:, self.grid.timeStep]
        phi = np.sum(self.grid.w[None, :, None] * fullTensor, axis=1)
        bbar = self.groupPlanck(T)
        sigma = self.material.sigma_a(self.grid.freqGrid, T)
        T_next = T + f * np.sum(sigma * (phi - bbar), axis=0)
        self.grid.temperatureSet[:, self.grid.timeStep] = T_next
        rhs = sigma * bbar
        return T_next, rhs

    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGrid, T) + self.timeAbsorption()


    def radiationSweep(self):
        self.startTimeStep()
        timeTerm = self.timeAbsorption()
        T_next, emission = self.materialEquation(self.grid.fullTensor)
        rhs = emission + timeTerm * self.psi_old
        mu = self.grid.muSet
        dx = self.grid.dx
        newfull = np.zeros_like(self.grid.fullTensor)
        phibl = self.boundaryCondition("left", self.grid.timeSet[self.grid.timeStep])
        phibr = self.boundaryCondition("right", self.grid.timeSet[self.grid.timeStep])
        sigStar = self.sigmaStar(T_next)

        for m, mu_val in enumerate(mu):
            if mu_val > 0:
                newfull[:, m, 0] = (rhs[:, 0] + (mu_val / dx) * phibl[:, m]) / (
                    mu_val / dx + sigStar[:, 0]
                )
                for i in range(self.params.nBins - 1):
                    newfull[:, m, i + 1] = (
                        rhs[:, i + 1] + (mu_val / dx) * self.grid.fullTensor[:, m, i]
                    ) / (mu_val / dx + sigStar[:, i + 1])
            else:
                newfull[:, m, -1] = (rhs[:, -1] + (abs(mu_val) / dx) * phibr[:, m]) / (
                    abs(mu_val) / dx + sigStar[:, -1]
                )
                for i in range(self.params.nBins - 1, 0, -1):
                    newfull[:, m, i - 1] = (
                        rhs[:, i - 1] + (abs(mu_val) / dx) * self.grid.fullTensor[:, m, i]
                    ) / (abs(mu_val) / dx + sigStar[:, i - 1])

        self.grid.fullTensor = newfull.copy()
        self.grid.temperatureSet[:, self.grid.timeStep] = T_next



class Marshak:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Equations(self.parameters, grid, self.material, constants)
        self.applyInitialConditions(grid)

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()
        