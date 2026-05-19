import numpy as np


# Non coupled equations
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
        if self.time_step == self.grid.timeStep:
            return

        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.fullTens = self.grid.fullTensor.copy()

    def radiationSweep(self):
        self.startTimeStep()

        timeTerm = self.timeAbsorption()  # Time derivative term
        mu = self.grid.muSet
        sig_tSet = self.material.sig_tAngle + timeTerm
        rhsfull = self.material.source(self.fullTens) + timeTerm * self.psi_old # shape (freq, sn, nBins)
        time = self.grid.timeSet[self.grid.timeStep]
        phibl = self.boundaryCondition("left", time)
        phibr = self.boundaryCondition("right", time)
        newFull = np.zeros_like(self.fullTens)

        for f in range(self.freq):
            rhs = rhsfull[f]
            phi = self.fullTens[f]
            new_phi = self.fullTens[f]
            sig_t = sig_tSet[f]


            for m in range(self.sn):
                if mu[m] > 0:
                    # Forward sweep
                    new_phi[m, 0] = (rhs[m, 0] + (mu[m] / self.grid.dx) * phibl[f, m]) / (mu[m] / self.grid.dx + sig_t[0])
                    for i in range(self.params.nBins-1):
                        new_phi[m, i + 1] = (
                            rhs[m, i+1] + (mu[m] / self.grid.dx) * phi[m, i]
                        ) / (mu[m] / self.grid.dx + sig_t[i+1])
                else:
                    # Backward sweep
                    new_phi[m, -1] = (rhs[m, -1] + (abs(mu[m]) / self.grid.dx) * phibr[f, m]) / (abs(mu[m]) / self.grid.dx + sig_t[-1])
                    for i in range(self.params.nBins - 1, 0, -1):
                        new_phi[m, i-1] = (
                            rhs[m, i-1] + (abs(mu[m]) / self.grid.dx) * phi[m, i]
                        ) / (abs(mu[m]) / self.grid.dx + sig_t[i-1])
            newFull[f] = new_phi
        self.grid.fullTensor = newFull.copy()

class CoupledEquations:
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


class MovingMeshEquations:
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

class Logic:
    def __init__(self, params, grid, material, constants):
        self.grid = grid
        self.constants = constants
        self.params = params
        self.material = material
    
        if not self.params.materialCoupled:
            self.equations = Equations(params, grid, material, constants)

        else:
            self.equations = CoupledEquations(params, grid, material, constants)

    def applyInitialConditions(self):
        self.equations.applyInitialConditions()

    def radiationSweep(self):
        self.equations.radiationSweep()

