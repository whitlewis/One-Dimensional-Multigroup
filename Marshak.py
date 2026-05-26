import numpy as np
import numpy.polynomial.legendre as leggauss
import Logic as Log

class Parameters:
    def __init__(self, maxIters=100, tol=1e-10, nSteps=1000, Transient=True):
        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol

        # Angular discretization parameters
        self.sn = 8

        # Initial and source temperature parameters
        self.initialTemperature = 0.5
        self.sourceTemp = 0.5

        # Spatial grid parameters
        self.xMin = -8
        self.xMax = 8
        self.nBins = 120

        # Boundary conditions (currently for all frequencies and angles, but could be replaced with planckian of temperature)
        self.boundaryLeft = 0.0
        self.boundaryRight = 0.0

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
    
    def sigma_a(self, freq, T): # Placeholder constant opacity
        return np.ones((self.params.freqNum, self.params.nBins))*100

    
    def C_v(self, T):  # Placeholder constant heat capacity
        return 1.0
    
    def addSource(self):
        # add a source somewhere in the domain
        source = np.zeros((self.grid.freqNum, self.grid.sn))
        source[:, :] = self.params.sourceTemp  # Set the source temperature for all frequencies and angles
        return source



class Marshak:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()
        