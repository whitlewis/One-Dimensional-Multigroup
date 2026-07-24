import numpy as np
import numpy.polynomial.legendre as leggauss
import Logic as Log
import Base as Base

class Parameters:
    def __init__(self, maxIters=200, tol=1e-15, nSteps=100, Transient=True):
        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol
        self.checkEnergy = True
        self.energyTol = 1e-15
        self.totalEnergy = 0.0

        # Angular discretization parameters
        self.sn = 4

        # Initial and source temperature parameters
        self.initialTemperature = 0.4       # material temperature
        self.radiationTemperature = 0.5     # Radiation temperature
        self.sourceTemp = 0.5

        # Spatial grid parameters
        self.xMin = -10
        self.xMax = 10
        self.nBins = 100

        # Boundary conditions (currently for all frequencies and angles, planckian at specified temperature or reflective)
        self.boundaryLeft = "Reflective"
        self.boundaryRight = "Reflective"

        # Group parameters
        self.freqNum = 25
        self.minFreq = 1e-4
        self.maxFreq = 30
        self.infFreq = 150

        # Time stepping parameters
        self.nSteps = nSteps
        self.timeMax = 1.0
        self.timeScale = "linear"  # "log" or "linear"

        # Choices of type of problem
        self.transient = Transient
        self.materialCoupled = True
        self.movingCoordinates = False
        self.energyCheckFreq = 200 # Check energy conservation every 200 time steps
        self.iterationCheck = False

class Material:
    def __init__(self, params, grid):
        self.params = params
        self.grid = grid
        self.const = Base.Constants()
    
    # Implementation of opacity from section 9.3 of McClarren's notes
    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
        return out


    # Planckian for opacity calculation
    def planckg(self):
        # Calculate the Planck function for each frequency group
        T = self.grid.temperatureSet[:, self.grid.timeStep]
        # FIX: Broadcast frequency as column (freqNum, 1) against T (nBins,) -> Result is (freqNum, nBins)
        nu_lo = self.grid.freqGrid[:-1, None] / T
        nu_hi = self.grid.freqGrid[1:, None] / T
        integrand = lambda nu: (15.0 * nu**3) / np.pi**4 /  np.expm1(nu)
        bg = self.simpson(integrand, nu_lo, nu_hi)
        return bg  # Shape is now (freqNum, nBins)
    
    def sigma_a(self, freq, T): 
        nu_lo = self.grid.freqGrid[:-1, None]
        nu_hi = self.grid.freqGrid[1:, None]
        sigma_aZero = np.ones((self.params.freqNum, self.params.nBins))
        denom = np.sqrt(T) * self.planckg()
        num = sigma_aZero * (np.exp(-nu_lo/T)-np.exp(-nu_hi/T))
        out = num / denom

        return out
    # End of opacity implementation
    
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
        