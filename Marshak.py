import numpy as np
import Logic as Log

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
        self.materialCoupled = True
        self.movingCoordinates = False

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



class Marshak:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

        