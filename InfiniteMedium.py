import numpy as np
import numpy.polynomial.legendre as leggauss
import Logic as Log

class Parameters:
    def __init__(self, maxIters=100, tol=1e-10, nSteps=1000, Transient=True):
        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol
        self.checkEnergy = True
        self.energyTol = 1e-15
        self.totalEnergy = 0.0

        # Angular discretization parameters
        self.sn = 8

        # Initial and source temperature parameters
        self.initialTemperature = 0.5       # material temperature
        self.radiationTemperature = 2.0     # Radiation temperature
        self.sourceTemp = 0.5

        # Spatial grid parameters
        self.xMin = -30
        self.xMax = 30
        self.nBins = 120

        # Boundary conditions (currently for all frequencies and angles, planckian of temperature)
        self.boundaryLeft = 0.5
        self.boundaryRight = 0.5

        # Group parameters
        self.freqNum = 45
        self.minFreq = 1e-3
        self.maxFreq = 30
        self.infFreq = 150

        # Time stepping parameters
        self.nSteps = nSteps
        self.timeMax = 0.2
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
        return .01
    



class InfiniteMedium:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()
        