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

