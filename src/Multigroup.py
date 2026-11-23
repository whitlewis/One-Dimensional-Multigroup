"""Standard multigroup (SM) and variable multigroup (VMG) problem setup.

Both methods share one Parameters class and one Material, so they always solve the same physical
problem with the same opacity and heat capacity. The only differences are the frequency-group
treatment (set by `method`) and where results are saved:

    SM  : fixed physical frequency groups
    VMG : groups in u = nu / scale(T) that move with the local temperature
"""
import numpy as np

import Base
import Logic as Log
import NickelOpacity

METHODS = ("SM", "VMG")
NI_DENSITY = 1.0  # g/cc; NickelOpacity loads only the rho = 1 g/cc TOPS table


class Parameters:
    def __init__(self, method="SM", tol=1e-8, maxIters=400, nSteps=400, Transient=True):
        method = method.upper()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method

        # Tolerance and iteration parameters
        self.maxIters = maxIters
        self.tol = tol
        self.checkEnergy = False
        self.energyTol = 1e-15
        self.totalEnergy = 0.0
        self.temperatureLearningRate = 1.0  # Learning rate for the VMG temperature update

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

        # Boundary conditions (all frequencies and angles; see Logic.BOUNDARY_TYPES)
        self.boundaryLeft = "Reflective"
        self.boundaryRight = "Reflective"
        self.setLeftBoundaryTemp = 0.5  # Temperature for Planckian, Bimodal or Delta boundary on the left
        self.setRightBoundaryTemp = 0.5  # Temperature for Planckian, Bimodal or Delta boundary on the right

        # Group parameters
        self.groupSpace = 'log' # log or linear
        self.freqNum = 100
        self.minFreq = 1e-8
        self.maxFreq = 35
        self.infFreq = 125
        self.matchNu = method == "VMG"  # VMG: divide u edges by T0 so they equal the SM edges initially

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
        self.movingCoordinates = method == "VMG"
        self.extrapolateTemp = False   # better first guess for T from the previous steps
        self.energyCheckFreq = 200     # Check energy conservation every 200 time steps
        self.iterationCheck = False
        self.andersonDepth = 5         # Anderson acceleration memory for the T iteration (0 = off)
        self.opacityModel = "nickel"   # "nickel" (TOPS Planck table) or "test" (1/(2T))
        self.useNumba = True           # numba sweeps (False = numpy/Python sweeps, identical results)
        self.runName = "Run"
        self.runLabel = "Run"
        self.fileFolder = method       # results go to datastash/sm or datastash/vmg
        self.saveResults = False       # False asks after the simulation


class Material:
    """Material properties shared by SM and VMG."""

    def __init__(self, params, grid):
        self.params = params
        self.grid = grid
        self.const = Base.Constants()

    def sigma_a(self, freq, T, frequencyScale=None):
        """Absorption opacity [1/cm], shape (freqNum, nBins), at the physical group centres.

        freq holds the group-centre coordinates. SM passes no frequencyScale: the groups are already
        physical. VMG passes its u -> nu scale (Logic.MovingMeshEquations.frequencyScale), and the
        opacity is evaluated at nu = u * scale, so both methods see the same sigma_a(nu, T).
        """
        temperature = np.broadcast_to(np.asarray(T, dtype=float), (self.params.nBins,))
        if getattr(self.params, "opacityModel", "nickel") == "test":
            return np.ones((self.params.freqNum, self.params.nBins)) * 1 / 2 / temperature  # 1/(2T) test opacity

        energy = np.asarray(freq, dtype=float)[:, None]
        if frequencyScale is not None:
            scale = np.broadcast_to(np.asarray(frequencyScale, dtype=float), (self.params.nBins,))
            energy = energy * scale[None, :]
        # Interpolated TOPS nickel Planck opacity, cm^2/g at rho = 1 g/cc -> sigma_a [1/cm] = kappa * rho
        return NI_DENSITY * NickelOpacity.interpolate_planck_opacity_pointwise(
            temperature[None, :], energy
        )

    def C_v(self, T):  # Placeholder constant heat capacity
        return .01


class Problem:
    """SM or VMG problem (chosen by params.method / params.movingCoordinates)."""

    def __init__(self, grid, constants, params):
        self.parameters = params
        self.material = Material(self.parameters, grid)
        self.equations = Log.Logic(self.parameters, grid, self.material, constants)
        self.equations.applyInitialConditions()

    def applyInitialConditions(self, grid):
        self.equations.applyInitialConditions()
