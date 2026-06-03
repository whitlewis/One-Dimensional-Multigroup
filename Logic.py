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
        fullTensorPhi = 2*np.pi*np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi

    # Simpson for integration over group
    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
        return out

    # Base Planck definiton
    def planck(self, nu, T):  # Planck function (not group integrated or weighted)
        denom = np.expm1(self.const.h * nu/T)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * nu**3 / denom

    # Function for initial Condition as Planckian (helper for initialCondition)  
    def initSpectra(self):
        T0 = self.params.radiationTemperature
        planck = self.planckBar(T0) 
        self.grid.fullTensor[:] = planck[:, None]
        return self.grid.fullTensor.copy()
    
    def energyInitialCondition(self):
        T0 = self.params.radiationTemperature
        groupEnergies = self.simpson(lambda nu: self.planck(nu, T0), self.grid.freqGrid[:-1], self.grid.freqGrid[1:])  # shape: (freqNum,)
        totalEnergy = np.sum(groupEnergies) * self.params.nBins  # Total energy density across all groups and spatial bins
        planckBarEnergy = np.sum(np.sum(self.getPhi() * self.grid.freqGroups[:, None]))  # Energy density from the group-averaged Planck function for reference
        if np.abs(totalEnergy - planckBarEnergy) > self.params.energyTol:
            print(f"⚠️ Initial energy check failed: Total energy density from initial condition ({totalEnergy:.4e}) does not match energy density from group-averaged Planck function ({planckBarEnergy:.4e})")
        EradExpected = self.params.nBins * self.const.a * self.const.c * T0**4  # Expected radiation energy density from the initial condition
        if np.abs(planckBarEnergy - EradExpected) > self.params.energyTol:
            print(f"⚠️ Initial energy check failed: Energy density from group-averaged Planck function ({planckBarEnergy:.4e}) does not match expected radiation energy density from initial condition ({EradExpected:.4e})")
        mult = EradExpected / planckBarEnergy  # Scaling factor to ensure correct initial energy density
        self.grid.fullTensor *= mult  # Scale the initial condition to ensure correct total energy density
        radTemp = (np.sum(np.sum(self.getPhi() * self.grid.freqGroups[:, None])) / self.params.nBins / self.const.a / self.const.c)**0.25
        print(f'Initial radiation temperature after scaling: {radTemp:.4e}')
        print(f'Planck init scaled by factor {mult:.4e} to ensure correct initial energy density with Radiation Temperature {T0:.4e}')

    
    # Set initial Conditions here
    def initialCondition(self):
        return self.initSpectra()
    
    # Helper function to define initial conditions
    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        self.energyInitialCondition()
        self.grid.fullTensorPhi = self.getPhi()
        self.fullTens = self.grid.fullTensor.copy()
        self.psi_old = self.grid.fullTensor.copy()


    # Helper for boundary
    def boundaryPlanck(self):
        if self.params.boundaryLeft == "Infinite":
            self.params.boundaryLeft = self.params.radiationTemperature
        if self.params.boundaryRight == "Infinite":
            self.params.boundaryRight = self.params.radiationTemperature
        T0 = self.params.boundaryLeft
        T1 = self.params.boundaryRight
        planckLeft = np.broadcast_to(self.planckBar(T0), (self.params.freqNum, self.sn))  # shape: (freqNum, sn)
        planckRight = np.broadcast_to(self.planckBar(T1), (self.params.freqNum, self.sn))  # shape: (freqNum, sn)
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


    # Group integrated Planck
    def planckBar(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar

    def sigmaBar(self):     # Placeholder for group-averaged opacity, currently unnecessary since we are using a constant opacity
        return
    
    def psiBar(self):       # placeholder, currently unnecessary due to init
        return

    # Definition of the coupled material equation (Correct now)
    def materialEquation(self):
        # Outer constant calc
        dt = self.grid.dt[self.grid.timeStep]
        f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
        
        # Lagged temperature and phi
        T = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature in all x cells (120,1)
        phi = self.getPhi()  # Compute scalar flux by integrating over angles
        bbar = self.planckBar(T)  # Get the group-averaged Planckian for the material 
        
        # Calculation of next temperature
        T_offset = f * np.sum((self.material.sigma_a(self.grid.freqGrid, T) * phi - 4*np.pi*self.material.sigma_a(self.grid.freqGrid, T) * bbar)* np.broadcast_to(self.grid.freqGroups[:,None], (self.params.freqNum, self.params.nBins)), axis=0)
        T_next = T + T_offset  # Update temperature using the material energy equation)
        # Calculation of the rhs of eq (Q*)
        bbarNext = self.planckBar(T)  # Get the group-averaged Planckian for the next temperature
        bbarNext = np.broadcast_to(bbarNext[:, None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(self.material.sigma_a(self.grid.freqGroups, T_next)[:,None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        bbar
        rhs = sa * bbarNext + self.timeTerm * self.psi_old # Right-hand side of the transport equation
        
        # Update grid object and persist next temperature on the equations object
        self.grid.rhs = rhs.copy()
        self.T_next = T_next.copy()
        self.grid.T_next = T_next.copy()



    # Define modified opacity
    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGrid, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity

    def radiationSweep(self):
        # Initialize time set assets
        self.startTimeStep()

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
            new_phi = np.zeros_like(self.fullTens[f])
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
            self.materialEquation = self.equations.materialEquation  # Expose material equation for coupled problems

    def applyInitialConditions(self):
        self.equations.applyInitialConditions()

    def radiationSweep(self):
        self.equations.radiationSweep()

    def getPhi(self):
        return self.equations.getPhi()

