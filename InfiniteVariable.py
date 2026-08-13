import numpy as np
import numpy.polynomial.legendre as leggauss
import Logic as Log
import Base as Base

class Parameters:
    def __init__(self, tol=1e-8, maxIters=200, nSteps=401, Transient=True):
        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol
        self.checkEnergy = False
        self.energyTol = 1e-6
        self.totalEnergy = 0.0
        self.temperatureLearningRate = 1.0  # Learning rate for temperature updates

        # Angular discretization parameters
        self.sn = 8

        # Initial and source temperature parameters
        self.initialTemperature = 0.4      # material temperature
        self.radiationTemperature = 0.5     # Radiation temperature
        self.sourceTemp = 0.5

        # Spatial grid parameters
        self.xMin = -1
        self.xMax = 1
        self.nBins = 100

        # Boundary conditions (currently for all frequencies and angles, planckian at specified temperature or reflective)
        self.boundaryLeft = "Reflective"
        self.boundaryRight = "Reflective"

        # Group parameters
        self.freqNum = 40
        self.minFreq = 1e-3
        self.maxFreq = 30
        self.infFreq = 150

        # Time stepping parameters
        self.nSteps = nSteps
        self.timeMax = 1.0
        self.timeScale = "log"  # "log" or "linear"

        # Choices of type of problem
        self.transient = Transient
        self.materialCoupled = True
        self.movingCoordinates = True
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
        nu_lo = self.grid.freqGrid[:-1, None]    # No need to divide by T here; we want the actual frequency range for for each group not the u range
        nu_hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: (15.0 * nu**3) / np.pi**4 /  np.expm1(nu)
        bg = self.simpson(integrand, nu_lo, nu_hi)
        return bg  # Shape is now (freqNum, nBins)
    
    def sigma_a(self, freq, T): 
        T = self.grid.temperatureSet[:, self.grid.timeStep]   # Get the nu from u grid
        nu_lo = self.grid.freqGrid[:-1, None] * T
        nu_hi = self.grid.freqGrid[1:, None] * T
        sigma_aZero = np.ones((self.params.freqNum, self.params.nBins))
        denom = np.sqrt(T) * self.planckg()
        num = sigma_aZero * (np.exp(-nu_lo/T)-np.exp(-nu_hi/T))
        out = np.clip(num / denom, a_min = 1.0e-8, a_max=1.0e9)  # Avoid division by zero and ensure non-negative opacities
        # print(f"Calculated opacities with shape: {out.shape}, min: {np.min(out):3e}, max: {np.max(out):3e}")  # Debugging statement
        return out

    
    def C_v(self, T):  # Placeholder constant heat capacity
        return .01
    



class InfiniteVariable:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()
        