import numpy as np
import numpy.polynomial.legendre as leggauss


class Parameters:
    def __init__(self, maxIters=1000, tol=1e-10, nSteps=1000):
        self.maxIters = maxIters
        self.tol = tol
        self.nSteps = nSteps

        self.nBins = 120
        self.xMin = -8
        self.xMax = 8

        self.sn = 8

        self.freqNum = 10
        self.timeMax = 20
        self.maxFreq = 150
        self.times = 1000


class Material:
    def __init__(self, params, grid):
        self.sig_a = np.zeros(params.freqNum)

        x = 0.5 * (grid.spaceGrid[:-1] + grid.spaceGrid[1:])

    def sigma_a(self, freq, T):
        return 100

    def C_v(self, T):
        return 1.0


class Equations:

    def __init__(self, params, grid, material, constants):
        self.params = params
        self.grid = grid
        self.material = material
        self.constants = constants

    def simpson(self, integrand, lo, hi):

        h = (hi - lo) / 3

        f0 = integrand(lo)
        f1 = integrand(lo + h)
        f2 = integrand(lo + 2*h)
        f3 = integrand(hi)

        return (3/8) * h[:, None] * (f0 + 3*f1 + 3*f2 + f3)

    def planck(self, nu, T, const):
        nu = nu[:, None]  # shape: (freqNum, 1)
        T = T[None, :]  # shape: (1, nBins)
        denom = np.expm1(nu / T)

        f = (15.0 * const.a * const.c) / (4.0 * np.pi**5)

        return f * nu**3 / denom

    def groupPlanck(self, T):

        lo = self.grid.freqGrid[:-1]
        hi = self.grid.freqGrid[1:]

        integrand = lambda nu: self.planck(nu, T, self.constants)

        bbar = self.simpson(integrand, lo, hi)

        return bbar

    def materialEquation(self, fullTensor):

        dt = self.grid.dt[self.grid.timeStep]

        T = self.grid.temperatureSet[:, self.grid.timeStep]

        f = dt / self.material.C_v(T)

        phi = np.sum(
            self.grid.w[None, :, None] * fullTensor,
            axis=1,
        )

        bbar = self.groupPlanck(T)

        sigma = self.material.sigma_a(self.grid.freqGrid[:-1, None], T[None, :])

        T_next = T + f * np.sum((sigma * phi - sigma * bbar), axis=0)

        self.grid.temperatureSet[:, self.grid.timeStep + 1] = T_next

        rhs = 4 * np.pi * sigma * bbar - sigma * phi

        return T_next, rhs

    def sigmaStar(self, freq, T):

        return (
            self.material.sigma_a(freq, T)
            + 1 / self.constants.c
            * 1 / self.grid.dt[self.grid.timeStep]
        )

    def radiationSweep(self, T):

        mu = self.grid.muSet

        phiBl = np.ones((self.params.sn, self.params.freqNum))

        phiBr = np.zeros((self.params.sn, self.params.freqNum))

        newfull = np.zeros_like(self.grid.fullTensor)

        T_next, rhs = self.materialEquation(self.grid.fullTensor)

        for f in range(self.params.freqNum):

            freq = self.grid.freqGrid[f]

            for m in range(self.params.sn):

                if mu[m] > 0:

                    newfull[f, m, 0] = (
                        rhs[f, 0]
                        + (mu[m] / self.grid.dx) * phiBl[m, f]
                    ) / (
                        mu[m] / self.grid.dx
                        + self.sigmaStar(freq, T_next[0])
                    )

                    for i in range(self.params.nBins - 1):

                        newfull[f, m, i + 1] = (
                            rhs[f, i + 1]
                            + (mu[m] / self.grid.dx)
                            * newfull[f, m, i]
                        ) / (
                            mu[m] / self.grid.dx
                            + self.sigmaStar(freq, T_next[i + 1])
                        )

                else:

                    newfull[f, m, -1] = (
                        rhs[f, -1]
                        + (abs(mu[m]) / self.grid.dx)
                        * phiBr[m, f]
                    ) / (
                        abs(mu[m]) / self.grid.dx
                        + self.sigmaStar(freq, T_next[-1])
                    )

                    for i in range(self.params.nBins - 1, 0, -1):

                        newfull[f, m, i - 1] = (
                            rhs[f, i - 1]
                            + (abs(mu[m]) / self.grid.dx)
                            * newfull[f, m, i]
                        ) / (
                            abs(mu[m]) / self.grid.dx
                            + self.sigmaStar(freq, T_next[i - 1])
                        )

        self.grid.fullTensor = newfull.copy()

        return newfull


class Marshak:
    def __init__(self, grid, constants):

        self.parameters = Parameters()

        self.material = Material(self.parameters, grid)

        self.equations = Equations(
            self.parameters,
            grid,
            self.material,
            constants,
        )
