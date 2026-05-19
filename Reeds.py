import numpy as np
import numpy.polynomial.legendre as leggauss
import Logic as Log
import warnings

class Parameters:
    def __init__(self, maxIters=100, tol=1e-10, nSteps=1000, Transient=True):
        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol

        # Angular discretization parameters
        self.sn = 8

        # Initial and source temperature parameters (not used in this example, but can be extended for thermal problems)
        self.initialTemperature = 0.0
        self.sourceTemp = 1.0

        # Spatial grid parameters
        self.xMin = -8
        self.xMax = 8
        self.nBins = 120

        # Group parameters
        self.freqNum = 1
        self.maxFreq = 150

        # Time stepping parameters
        self.nSteps = nSteps
        self.timeMax = 20.0
        self.timeScale = "linear"  # "log" or "linear"

        # Choices of type of problem
        self.transient = Transient
        self.materialCoupled = False
        self.movingCoordinates = False


class Material:
    def __init__(self, params, grid):
        self.Grid = grid
        self.params = params

        n = params.nBins
        x = 0.5 * (grid.spaceGrid[:-1] + grid.spaceGrid[1:])  # cell centers
        self.sig_a = np.zeros(n)
        self.sig_s = np.zeros(n)
        self.sig_t = np.zeros(n)
        self.Q = np.zeros((params.freqNum, n))
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
        self.Q[:, mat1] = 50.0  # Q can be change per Group (thanks Johannes)
        self.Q[:, mat3] = 1.0

        # Scattering
        self.sig_s[mat3] = 0.9
        self.sig_s[mat4] = 0.9

        self.sig_t = self.sig_a + self.sig_s

        self.sig_aAngle = np.tile(self.sig_a, (params.freqNum, 1))
        self.sig_sAngle = np.tile(self.sig_s, (params.freqNum, 1))
        self.sig_tAngle = np.tile(self.sig_t, (params.freqNum, 1))
        self.QAngle = np.tile(self.Q, (params.freqNum, 1))

    def source(self, fullTensor):
        phig = np.sum(fullTensor * self.Grid.w[None, :, None], axis=1) / np.sum(self.Grid.w)  # (nfreq, nBins)
        rhs = self.sig_sAngle * phig + self.Q  # (nfreq, nBins)        
        return np.broadcast_to(rhs[:, None, :], fullTensor.shape).copy()


class Reeds:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()        

        