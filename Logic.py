import numpy as np


# Non coupled equations
class Equations:

    def __init__(self, params, grid, material, constants):
        # Add passed classes
        self.params = params
        self.grid = grid
        self.material = material
        self.const = constants

        # Init tensor for calculation
        self.fullTens = grid.fullTensor.copy()  # shape: (freqNum, sn, nBins)
        self.mu = self.grid.muSet

        # Init parameters
        self.freq = params.freqNum
        self.sn = params.sn
        self.dx = grid.dx
        self.time_step = None
        self.timeTerm = 0.0


    # Define initial conditions here
    def initialCondition(self):
        return np.zeros_like(self.grid.fullTensor)
    
    # Helper function to define initial conditions
    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        self.fullTens = self.grid.fullTensor.copy()
        self.psi_old = self.grid.fullTensor.copy()

    # Possible time varying Boundary condition
    def boundaryCondition(self, side, time):
        return np.zeros((self.freq, self.sn))

    # Time term from discretization
    def timeAbsorption(self):
        if not self.params.transient:
            return 0.0
        return 1.0 / (self.const.c * self.grid.dt[self.grid.timeStep])

    # Start the step
    def startTimeStep(self):
        if self.time_step == self.grid.timeStep:
            return

        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.fullTens = self.grid.fullTensor.copy()
        self.timeTerm = self.timeAbsorption()

    def radiationSweep(self):
        # Initialize time set assets
        self.startTimeStep()

        # Add time term to sigma* and Q*
        sig_tSet = self.material.sig_tAngle + self.timeTerm
        rhsfull = self.material.source(self.fullTens) + self.timeTerm * self.psi_old # shape (freq, sn, nBins)

        # Set up Boundary conditions at the time step (allows for variable boundary conditions)
        time = self.grid.timeSet[self.grid.timeStep]
        phibl = self.boundaryCondition("left", time)
        phibr = self.boundaryCondition("right", time)

        # Initialize the next tensor
        newFull = np.zeros_like(self.fullTens)

        # Loop through frequency
        for f in range(self.freq):
            rhs = rhsfull[f]
            phi = self.fullTens[f]
            new_phi = self.fullTens[f]
            sig_t = sig_tSet[f]

            # Loop through Angles
            for m in range(self.sn):
                # Upwind 
                if self.mu[m] > 0:
                    # Forward sweep
                    new_phi[m, 0] = (rhs[m, 0] + (self.mu[m] / self.grid.dx) * phibl[f, m]) / (self.mu[m] / self.grid.dx + sig_t[0])
                    for i in range(self.params.nBins-1):
                        new_phi[m, i + 1] = (
                            rhs[m, i+1] + (self.mu[m] / self.grid.dx) * phi[m, i]
                        ) / (self.mu[m] / self.grid.dx + sig_t[i+1])

                else:
                    # Backward sweep
                    new_phi[m, -1] = (rhs[m, -1] + (abs(self.mu[m]) / self.grid.dx) * phibr[f, m]) / (abs(self.mu[m]) / self.grid.dx + sig_t[-1])
                    for i in range(self.params.nBins - 1, 0, -1):
                        new_phi[m, i-1] = (
                            rhs[m, i-1] + (abs(self.mu[m]) / self.grid.dx) * phi[m, i]
                        ) / (abs(self.mu[m]) / self.grid.dx + sig_t[i-1])
            # for each group update the tensor
            newFull[f] = new_phi

        # Set fullTensor to the updated Tensor
        self.grid.fullTensor = newFull.copy()

class CoupledEquations:
    def __init__(self, params, grid, material, constants):
        # Init classes
        self.params = params
        self.grid = grid
        self.material = material
        self.const = constants

        # Init calculation helper parameters
        self.freq = params.freqNum
        self.sn = params.sn
        self.dx = grid.dx
        self.time_step = None
        self.timeTerm = 0.0

        # Init tensors for calculations
        self.fullTensor = grid.fullTensor.copy()  # shape: (freqNum, sn, nBins)
        self.mu = self.grid.muSet
        self.rhs = np.zeros((self.freq, self.params.nBins))
        self.T_next = np.ones(self.params.nBins)*self.params.initialTemperature
    
    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi

    # Simpson for integration over group
    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
        return out

    # Base Planck definiton
    def planck(self, nu, T):  # Planck function (not group integrated or weighted)
        denom = np.expm1(nu/T)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * nu**3 / denom

    # Group integrated Planck
    def groupPlanck(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar

    # Function for initial Condition as Planckian (helper for initialCondition)  
    def initSpectra(self):
        T0 = self.params.radiationTemperature
        planck = self.groupPlanck(T0) 
        print(f' initial shape check: {planck.shape}')  # Check the shape of the planck function
        self.grid.fullTensor[:] = planck[:, None]
        self.grid.updateFullTensor(self.grid.fullTensor)
        return self.grid.fullTensor.copy()

     # Define initial conditions here
    
    # Set initial Conditions here
    def initialCondition(self):
        return self.initSpectra()
    
    # Helper function to define initial conditions
    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        self.grid.fullTensorPhi = self.getPhi()
        self.fullTens = self.grid.fullTensor.copy()
        self.psi_old = self.grid.fullTensor.copy()


    # Helper for boundary
    def boundaryPlanck(self):
        T0 = self.params.boundaryLeft
        T1 = self.params.boundaryRight
        planckLeft = np.broadcast_to(self.groupPlanck(T0), (self.params.freqNum, self.sn))  # shape: (freqNum, sn)
        planckRight = np.broadcast_to(self.groupPlanck(T1), (self.params.freqNum, self.sn))  # shape: (freqNum, sn)
        return planckLeft, planckRight

    # Possible time varying Boundary condition
    def boundaryCondition(self, side, time):
        left, right = self.boundaryPlanck()
        if side == "left":
            return left
        elif side == "right":
            return right

    # Time term from discretization
    def timeAbsorption(self):
        if not self.params.transient:
            return 0.0
        return 1.0 / (self.const.c * self.grid.dt[self.grid.timeStep])

    # Start the step
    def startTimeStep(self):
        if self.time_step == self.grid.timeStep:
            return

        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.fullTens = self.grid.fullTensor.copy()
        self.timeTerm = self.timeAbsorption()
        self.sigmaStarVar = self.sigmaStar(self.grid.temperatureSet[:, self.grid.timeStep])  # Update sigma* for the time step

    def sigmaGroup(self):
        return
    
    def phiGroup(self):
        return

    # Definition of the coupled material equation (Correct now)
    def materialEquation(self):
        # Outer constant calc
        dt = self.grid.dt[self.grid.timeStep]
        f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
        
        # Lagged temperature and phi
        T = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature in all x cells (120,1)
        phi = self.getPhi()  # Compute scalar flux by integrating over angles
        bbar = self.groupPlanck(T)  # Get the group-averaged Planckian for the material 

        # Calculation of next temperature
        T_next = T + f * np.sum((self.material.sigma_a(self.grid.freqGrid, T) * phi - 4*np.pi*self.material.sigma_a(self.grid.freqGrid, T) * bbar), axis=0)  # Update temperature using the material energy equation)

        # Calculation of the rhs of eq (Q*)
        rhs = 4 * np.pi * self.material.sigma_a(self.grid.freqGroups, T_next) * bbar + self.timeTerm * self.psi_old # Right-hand side of the transport equation

        # Update grid object
        self.grid.rhs = rhs
        self.grid.T_next = T_next
    
    # Define modified opacity
    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGrid, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity

    def radiationSweep(self):
        # Initialize time set assets
        self.startTimeStep()

        # update Temperature and get Q*
        self.materialEquation()

        # Set up Boundary conditions at the time step (allows for variable boundary conditions)
        time = self.grid.timeSet[self.grid.timeStep]
        phibl = self.boundaryCondition("left", time)
        phibr = self.boundaryCondition("right", time)

        # Initialize the next tensor
        newFull = np.zeros_like(self.fullTens)

        # Loop through frequency
        for f in range(self.freq):
            rhs = self.grid.rhs[f]
            phi = self.fullTens[f]
            new_phi = self.fullTens[f]
            sig_t = self.sigmaStarVar[f]

            # Loop through Angles
            for m in range(self.sn):
                # Upwind 
                if self.mu[m] > 0:
                    # Forward sweep
                    new_phi[m, 0] = (rhs[m, 0] + (self.mu[m] / self.grid.dx) * phibl[f, m]) / (self.mu[m] / self.grid.dx + sig_t[0])
                    for i in range(self.params.nBins-1):
                        new_phi[m, i + 1] = (
                            rhs[m, i+1] + (self.mu[m] / self.grid.dx) * phi[m, i]
                        ) / (self.mu[m] / self.grid.dx + sig_t[i+1])

                else:
                    # Backward sweep
                    new_phi[m, -1] = (rhs[m, -1] + (abs(self.mu[m]) / self.grid.dx) * phibr[f, m]) / (abs(self.mu[m]) / self.grid.dx + sig_t[-1])
                    for i in range(self.params.nBins - 1, 0, -1):
                        new_phi[m, i-1] = (
                            rhs[m, i-1] + (abs(self.mu[m]) / self.grid.dx) * phi[m, i]
                        ) / (abs(self.mu[m]) / self.grid.dx + sig_t[i-1])
            # for each group update the tensor
            newFull[f] = new_phi

        # Set fullTensor to the updated Tensor
        self.grid.fullTensor = newFull.copy()


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

