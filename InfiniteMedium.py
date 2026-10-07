import numpy as np
import numpy.polynomial.legendre as leggauss
import Logic as Log
import Base as Base

class Parameters:
    def __init__(self, maxIters=400, tol=1e-8, nSteps=400, Transient=True):
        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol
        self.checkEnergy = False
        self.energyTol = 1e-15
        self.totalEnergy = 0.0

        # Angular discretization parameters
        self.sn = 8

        # Initial and source temperature parameters
        self.initialTemperature = 0.4       # material temperature
        self.radiationTemperature = 0.5     # Radiation temperature
        self.sourceTemp = 0.5
        self.colorTemperature = 0.5

        # Spatial grid parameters
        self.xMin = -1
        self.xMax = 1
        self.nBins = 100

        # Boundary conditions (currently for all frequencies and angles, planckian at specified temperature or reflective)
        self.boundaryLeft = "Reflective"
        self.boundaryRight = "Reflective"
        self.setLeftBoundaryTemp = 0.5  # Temperature for Planckian or delta boundary condition on the left
        self.setRightBoundaryTemp = 0.5  # Temperature for Planckian or delta boundary condition on the right

        # Group parameters
        self.groupSpace = 'log' # log or linear
        self.freqNum = 100
        self.minFreq = 1e-8
        self.maxFreq = 35
        self.infFreq = 125

        # Time stepping parameters
        self.nSteps = nSteps
        self.timeMax = 0.1
        self.timeScale = "log"  # "log" or "linear"
        self.logLinTime = "log"
        self.stepSplit = .5
        self.splitStepsBool = False
        self.timeSplit = 0.1

        # Choices of type of problem
        self.transient = Transient
        self.materialCoupled = True
        self.movingCoordinates = False
        self.energyCheckFreq = 200 # Check energy conservation every 200 time steps
        self.iterationCheck = False
        self.fileFolder = "InfiniteMedium"
        self.runName = "Run"
        self.saveResults = False  # Flag to determine whether to save results after simulation (asks after the simulation)

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

    def planck(self, nu, T):  # Planck function (not group integrated or weighted)
        denom = np.expm1(self.const.h * nu/T)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * nu**3 / denom

    def sigmaAP(self, nu, T):
        sa0 = 10
        num = 1 - np.exp(-nu / T)
        denom = nu**3 * T*(1/2)
        return sa0 * num / denom

    # Planckian for opacity calculation
    def planckg(self, nu_lo, nu_hi):
        # Calculate the Planck function for each frequency group
        T = self.grid.temperatureSet[:, self.grid.timeStep]
        # FIX: Broadcast frequency as column (freqNum, 1) against T (nBins,) -> Result is (freqNum, nBins)
        nu_lo = nu_lo / T
        nu_hi = nu_hi / T
        integrand = lambda nu: (15.0 * nu**3) / np.pi**4 /  np.expm1(nu)
        bg = self.simpson(integrand, nu_lo, nu_hi)
        return bg  # Shape is now (freqNum, nBins)
    
    # def sigma_a(self, freq, T): 
    #     nu_lo = self.grid.freqGrid[:-1, None]
    #     nu_hi = self.grid.freqGrid[1:, None]
    #     sigma_aZero = 10 * np.ones((self.params.freqNum, self.params.nBins))
    #     denom = np.sqrt(T) * self.planckg(nu_lo, nu_hi)
    #     num = sigma_aZero * (np.exp(-nu_lo/T)-np.exp(-nu_hi/T))
    #     out = np.clip(num / denom, a_min=1e-8, a_max=1e10)
    #     # out = np.ones((self.params.freqNum, self.params.nBins))  # For testing purposes, set all opacities to a constant value
    #     return out

    def sigma_a(self, freq, T):
        nu_lo = self.grid.freqGrid[:-1, None]
        nu_hi = self.grid.freqGrid[1:, None]
        num = lambda nu: self.planck(nu, T) * self.sigmaAP(nu, T)
        denom = lambda nu: self.planck(nu, T)
        numG = self.simpson(num, nu_lo, nu_hi)
        denomG = self.simpson(denom, nu_lo, nu_hi)
        out = numG / denomG
        return out
    # End of opacity implementation
    
    def C_v(self, T):  # Placeholder constant heat capacity
        return .01
    



class InfiniteMedium:
    def __init__(self, grid, constants, params):
        self.parameters = params
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()
        