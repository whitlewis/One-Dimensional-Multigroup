import numpy as np
import numpy.polynomial.legendre as leggauss
import warnings

class Parameters:
    def __init__(self, maxIters=500, tol=1e-10, nSteps=100):
        self.maxIters = maxIters
        self.tol = tol
        self.nSteps = nSteps
        self.nBins = 120
        self.xMin = -8
        self.xMax = 8
        self.sn = 8
        self.freqNum = 1
        self.timeMax = 20
        self.maxFreq = 150
        self.initialTemperature = 0.0
        self.sourceTemp = 1.0


class Material:
    def __init__(self, params, grid):
        self.Grid = grid
        self.params = params

        n = params.nBins
        x = 0.5 * (grid.spaceGrid[:-1] + grid.spaceGrid[1:])  # cell centers
        self.sig_a = np.zeros(n)
        self.sig_s = np.zeros(n)
        self.sig_t = np.zeros(n)
        self.Q = np.zeros(n)
        self.Cv = np.ones(n)*.1
        self.freq = params.freqNum
        
        # Masks
        mat1 = (x >= -2) & (x <= 2)
        mat2 = ((x > 2) & (x <= 3)) | ((x < -2) & (x >= -3))
        vac = ((x > 3) & (x <= 5)) | ((x < -3) & (x >= -5)) | (x > 8) | (x < -8)
        mat3 = ((x > 5) & (x <= 6)) | ((x < -5) & (x >= -6))
        mat4 = ((x > 6) & (x <= 8)) | ((x < -6) & (x >= -8))

        # Absorption
        self.sig_a[mat1] = 50.0
        self.sig_a[mat2] = 5.0
        self.sig_a[mat3] = 0.1
        self.sig_a[mat4] = 0.1
        self.sig_a[vac] = 0.0

        # Constant source
        self.Q[mat1] = 50.0
        self.Q[mat3] = 1.0

        # Scattering
        self.sig_s[mat3] = 0.9
        self.sig_s[mat4] = 0.9

        self.sig_t = self.sig_a + self.sig_s

        self.sig_aAngle = np.tile(self.sig_a, (params.freqNum, 1))
        self.sig_sAngle = np.tile(self.sig_s, (params.freqNum, 1))
        self.sig_tAngle = np.tile(self.sig_t, (params.freqNum, 1))
        self.QAngle = np.tile(self.Q, (params.freqNum, 1))

    def source(self, fullTensor):
        interactionMatrix = np.diag(np.ones(self.params.freqNum))
        phig = np.sum(fullTensor * self.Grid.w[None, :, None], axis=1) / np.sum(self.Grid.w)  # (nfreq, nBins)
        scattered = interactionMatrix @ phig  # (nfreq, nBins)
        scattered = scattered[:, None, :] * self.sig_sAngle[:, :, None]  # (nfreq, sn, nBins)
        rhs = scattered + self.QAngle[:, None, :]  # (nfreq, sn, nBins)
        return rhs


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


    def radiationSweep(self):
        timeTerm = 1.0 / self.const.c / self.grid.dt[self.grid.timeStep]  # Time derivative term
        mu = self.grid.muSet
        sig_tSet = self.material.sig_tAngle + timeTerm
        sig_aSet = self.material.sig_aAngle + timeTerm
        sig_sSet = self.material.sig_sAngle
        rhsfull = self.grid.rhsfull  # shape (freq, sn, nBins)
        phibl = np.zeros(self.params.sn)  # Boundary condition: zero incoming flux
        phibr = np.zeros(self.params.sn)  # Boundary condition: zero incoming flux
        newFull = np.zeros_like(self.fullTens)

        for f in range(self.freq):
            rhs = rhsfull[f]
            phi = self.fullTens[f]
            new_phi = self.fullTens[f]
            sig_a = sig_aSet[f]
            sig_t = sig_tSet[f]
            sig_s = sig_sSet[f]

            for m in range(self.sn):
                if mu[m] > 0:
                    # Forward sweep
                    new_phi[m, 0] = (rhs[m, 0] + (mu[m] / self.grid.dx) * phibl[m]) / (mu[m] / self.grid.dx + sig_a[0])
                    for i in range(self.params.nBins-1):
                        new_phi[m, i + 1] = (
                            rhs[m, i+1] + (mu[m] / self.grid.dx) * phi[m, i]
                        ) / (mu[m] / self.grid.dx + sig_t[i+1])
                else:
                    # Backward sweep
                    new_phi[m, -1] = (rhs[m, -1] + (abs(mu[m]) / self.grid.dx) * phibr[m]) / (abs(mu[m]) / self.grid.dx + sig_a[-1])
                    for i in range(self.params.nBins - 1, 0, -1):
                        new_phi[m, i-1] = (
                            rhs[m, i-1] + (abs(mu[m]) / self.grid.dx) * phi[m, i]
                        ) / (abs(mu[m]) / self.grid.dx + sig_t[i-1])
            newFull[f] = new_phi
        self.grid.fullTensor = newFull.copy()
        return self.grid, newFull, 0  # Placeholder for T_next, (not needed for this problem)

class Reeds:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Equations(self.parameters, grid, self.material, constants)

        