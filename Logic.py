import numpy as np
np.seterr(divide='raise', invalid='raise', over='raise')
from numba import njit

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
    
    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = 4*np.pi*np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi

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
                            rhs[m, i+1] + (self.mu[m] / self.grid.dx) * new_phi[m, i]
                        ) / (self.mu[m] / self.grid.dx + sig_t[i+1])

                else:
                    # Backward sweep
                    new_phi[m, -1] = (rhs[m, -1] + (abs(self.mu[m]) / self.grid.dx) * phibr[f, m]) / (abs(self.mu[m]) / self.grid.dx + sig_t[-1])
                    for i in range(self.params.nBins - 1, 0, -1):
                        new_phi[m, i-1] = (
                            rhs[m, i-1] + (abs(self.mu[m]) / self.grid.dx) * new_phi[m, i]
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
        self.reflMatrix = [np.argmin(np.abs(self.grid.muSet + muVal)) for muVal in self.grid.muSet]  # Precompute reflection matrix for efficiency

        # Init tensors for calculations
        self.fullTensor = grid.fullTensor.copy()  # shape: (freqNum, sn, nBins)
        self.mu = self.grid.muSet
        self.rhs = np.zeros((self.freq, self.params.nBins))
        self.T_next = np.ones(self.params.nBins)*self.params.initialTemperature
    
    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = 4*np.pi*np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
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
        groupEnergies = self.simpson(lambda nu: self.planck(nu, T0), self.grid.freqGrid[:-1], self.grid.freqGrid[1:])
        
        # Calculate the exact expected macroscopic density per cell
        EradExpectedPerCell = self.const.a * T0**4
        EradExpected = self.params.nBins * EradExpectedPerCell
        
        # Calculate what your discrete phi tensor currently reads per cell
        currentPhiPerCell = np.sum(self.getPhi()) / self.params.nBins
        planckBarEnergyPerCell = currentPhiPerCell / self.const.c
        planckBarEnergy = planckBarEnergyPerCell * self.params.nBins
        
        # Calculate total energy directly from the analytical group sum per cell
        totalEnergy = (4.0 * np.pi / self.const.c) * np.sum(groupEnergies) * self.params.nBins
        
        if np.abs(totalEnergy - planckBarEnergy) > self.params.energyTol:
            print(f"⚠️ Initial energy check failed: Total energy density from initial condition ({totalEnergy:.4e}) does not match energy density from group-averaged Planck function ({planckBarEnergy:.4e})")
            
        if np.abs(planckBarEnergy - EradExpected) > self.params.energyTol:
            print(f"⚠️ Initial energy check failed: Energy density from group-averaged Planck function ({planckBarEnergy:.4e}) does not match expected radiation energy density from initial condition ({EradExpected:.4e})")
            
        # This force-scales your angular intensity tensor to perfectly match a*T^4 per cell
        mult = EradExpected / planckBarEnergy
        self.grid.fullTensor *= mult
        
        total_u = np.sum(self.getPhi()) / self.const.c
        radTemp = (total_u / self.params.nBins / self.const.a)**0.25
        
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

        T0 = self.params.setLeftBoundaryTemp if self.params.boundaryLeft in ["Infinite", "Reflective", "Vacuum"] else self.params.setLeftBoundaryTemp
        T1 = self.params.setRightBoundaryTemp if self.params.boundaryRight in ["Infinite", "Reflective", "Vacuum"] else self.params.setRightBoundaryTemp
        planckLeft = np.broadcast_to(self.planckBar(T0), (self.params.freqNum, self.sn)).copy()  # shape: (freqNum, sn)
        planckRight = np.broadcast_to(self.planckBar(T1), (self.params.freqNum, self.sn)).copy()  # shape: (freqNum, sn)
        if self.params.boundaryLeft == "Delta":
            totalFlux = np.sum(self.grid.du * self.simpson(lambda nu: self.planck(nu, T0), self.grid.freqGrid[:-1], self.grid.freqGrid[1:]))
            planckLeft[self.params.freqNum//4, :] = totalFlux / self.grid.du[self.params.freqNum//4]  # Delta function at the middle frequency group
        if self.params.boundaryRight == "Delta":
            totalFlux = np.sum(self.grid.du * self.simpson(lambda nu: self.planck(nu, T1), self.grid.freqGrid[:-1], self.grid.freqGrid[1:]))
            planckRight[self.params.freqNum//4, :] = totalFlux / self.grid.du[self.params.freqNum//4]      # Delta function at the middle frequency group
        if self.params.boundaryLeft not in ["Infinite", "Reflective", "Vacuum", "Delta", "Planckian"]:
            raise ValueError("Boundary must be one of: [Infinite, Reflective, Vacuum, Delta, Planckian]")
        if self.params.boundaryRight not in ["Infinite", "Reflective", "Vacuum", "Delta", "Planckian"]:
            raise ValueError("Boundary must be one of: [Infinite, Reflective, Vacuum, Delta, Planckian]")
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
        self.sigmaStarVar = self.sigmaStar(self.grid.T_next)  # Update sigma* for the time step


    # Group integrated Planck
    def planckBar(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar

    def sigmaBar(self, T):     # Placeholder for group-averaged opacity, currently unnecessary since we are using a constant opacity
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T)
        sbar = self.simpson(integrand, lo, hi)
        return sbar
    
    def psiBar(self):       # placeholder, currently unnecessary due to init
        return

    # Definition of the coupled material equation (Correct now)
    def materialEquation(self):
        # Outer constant calc
        dt = self.grid.dt[self.grid.timeStep]
        f =  dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
        
        # Lagged temperature and phi
        T = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature in all x cells (120,1)
        T_iterative = self.grid.T_next
        phi = self.getPhi()  # Compute scalar flux by integrating over angles
        bbar = self.planckBar(T_iterative)  # Get the group-averaged Planckian for the material 
        
        # Calculation of next temperature
        T_offset = f * np.sum((self.material.sigma_a(self.grid.freqGroups, T_iterative) * phi - 4*np.pi*self.material.sigma_a(self.grid.freqGroups, T_iterative) * bbar), axis=0)
        T_next = T + T_offset  # Update temperature using the material energy equation)
        self.T_next = T_next.copy()
        self.grid.T_next = T_next.copy()

    def rhsUpdate(self):
        T = self.grid.T_next
        bbarNext = self.planckBar(T)  # Get the group-averaged Planckian for the next temperature
        bbarNext = np.broadcast_to(bbarNext[:, None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(self.material.sigma_a(self.grid.freqGroups, T)[:,None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        rhs = sa * bbarNext + self.timeTerm * self.psi_old
        self.grid.rhs = rhs.copy()
        self.sigmaStarVar = self.sigmaStar(self.grid.T_next)


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
                    if self.params.boundaryLeft == "Reflective":
                        # Reflective BC: use the reflected angle's outgoing flux as the incoming flux
                        reflected_m = self.reflMatrix[m]
                        bVal = new_phi[reflected_m, 0]  # Use the reflected angle's flux at the boundary
                    elif self.params.boundaryLeft == "Vacuum":
                        bVal = 0.0  # Vacuum boundary condition: incoming flux is zero
                    else:
                        bVal = phibl[f, m]  # Use the specified boundary condition if not reflective
                    
                    new_phi[m, 0] = (rhs[m, 0] + (self.mu[m] / self.grid.dx) * bVal) / (self.mu[m] / self.grid.dx + sig_t[0])
                    for i in range(self.params.nBins - 1):
                        new_phi[m, i + 1] = (
                            rhs[m, i+1] + (self.mu[m] / self.grid.dx) * new_phi[m, i]
                        ) / (self.mu[m] / self.grid.dx + sig_t[i+1])

                else:
                    # Backward sweep

                    if self.params.boundaryRight == "Reflective":
                        # Reflective BC: use the reflected angle's outgoing flux as the incoming flux
                        reflected_m = self.reflMatrix[m]
                        bVal = self.fullTens[f, reflected_m, -1]  # Use the reflected angle's flux at the boundary
                    elif self.params.boundaryRight == "Vacuum":
                        bVal = 0.0  # Vacuum boundary condition: incoming flux is zero
                    else:
                        bVal = phibr[f, m]  # Use the specified boundary condition if not reflective

                    new_phi[m, -1] = (rhs[m, -1] + (abs(self.mu[m]) / self.grid.dx) * bVal) / (abs(self.mu[m]) / self.grid.dx + sig_t[-1])
                    for i in range(self.params.nBins - 1, 0, -1):
                        new_phi[m, i-1] = (
                            rhs[m, i-1] + (abs(self.mu[m]) / self.grid.dx) * new_phi[m, i]
                        ) / (abs(self.mu[m]) / self.grid.dx + sig_t[i-1])
            # for each group update the tensor
            newFull[f] = new_phi

        # Set fullTensor to the updated Tensor
        self.grid.fullTensor = newFull.copy()



@njit
def planckBarNumbaVariable(freqGrid, T, a, c, h):
    nGroups = freqGrid.shape[0] - 1
    nBins = T.shape[0]

    out = np.empty((nGroups, nBins))

    C = (15.0 * a * c) / (4.0 * np.pi**5)

    for g in range(nGroups):
        lo = freqGrid[g]
        hi = freqGrid[g + 1]
        dx = (hi - lo) / 3.0

        u0 = lo
        u1 = lo + dx
        u2 = lo + 2.0 * dx
        u3 = hi

        for i in range(nBins):
            Ti = T[i]
            T4 = Ti**4

            B0 = C * u0**3 * T4 / np.expm1(h * u0)
            B1 = C * u1**3 * T4 / np.expm1(h * u1)
            B2 = C * u2**3 * T4 / np.expm1(h * u2)
            B3 = C * u3**3 * T4 / np.expm1(h * u3)

            out[g, i] = (3.0 / 8.0) * dx * (
                B0 + 3.0*B1 + 3.0*B2 + B3
            )

    return out

class MovingMeshEquations:
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
        self.reflMatrix = [np.argmin(np.abs(self.grid.muSet + muVal)) for muVal in self.grid.muSet]  # Precompute reflection matrix

        # Init tensors for calculations
        self.fullTensor = grid.fullTensor.copy()  # shape: (freqNum, sn, nBins)
        self.mu = self.grid.muSet
        self.rhs = np.zeros((self.freq, self.params.nBins))
        self.T_next = np.ones(self.params.nBins)*self.params.initialTemperature

    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = 4*np.pi*np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi

    # Simpson for integration over group
    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
        return out

    # Base Planck definiton
    def planck(self, u, T):  # Planck function for variable basis (not group integrated or weighted)
        denom = np.expm1(self.const.h * u)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * u**3 * T**4 / denom

    # Group integrated Planck
    def planckBarInit(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda u: self.planck(u, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar
    
    def planckBar(self, T):
        return planckBarNumbaVariable(
            self.grid.freqGrid,
            T,
            self.const.a,
            self.const.c,
            self.const.h
        )

    # Function for initial Condition as Planckian (helper for initialCondition)  
    def initSpectra(self):
        T0 = self.params.radiationTemperature
        planck = self.planckBarInit(T0) 
        self.grid.fullTensor[:] = planck[:, None]
        return self.grid.fullTensor.copy()
    
    def energyInitialCondition(self):
        T0 = self.params.radiationTemperature
        groupEnergies = self.simpson(lambda nu: self.planck(nu, T0), self.grid.freqGrid[:-1], self.grid.freqGrid[1:])
        
        # Calculate the exact expected macroscopic density per cell
        EradExpectedPerCell = self.const.a * T0**4
        EradExpected = self.params.nBins * EradExpectedPerCell
        
        # Calculate what your discrete phi tensor currently reads per cell
        currentPhiPerCell = np.sum(self.getPhi()) / self.params.nBins
        planckBarEnergyPerCell = currentPhiPerCell / self.const.c
        planckBarEnergy = planckBarEnergyPerCell * self.params.nBins
        
        # Calculate total energy directly from the analytical group sum per cell
        totalEnergy = (4.0 * np.pi / self.const.c) * np.sum(groupEnergies) * self.params.nBins
        
        if np.abs(totalEnergy - planckBarEnergy) > self.params.energyTol:
            print(f"⚠️ Initial energy check failed: Total energy density from initial condition ({totalEnergy:.4e}) does not match energy density from group-averaged Planck function ({planckBarEnergy:.4e})")
            
        if np.abs(planckBarEnergy - EradExpected) > self.params.energyTol:
            print(f"⚠️ Initial energy check failed: Energy density from group-averaged Planck function ({planckBarEnergy:.4e}) does not match expected radiation energy density from initial condition ({EradExpected:.4e})")
            
        # This force-scales your angular intensity tensor to perfectly match a*T^4 per cell
        mult = EradExpected / planckBarEnergy
        self.grid.fullTensor *= mult
        
        total_u = np.sum(self.getPhi()) / self.const.c
        radTemp = (total_u / self.params.nBins / self.const.a)**0.25
        
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
        T0 = self.params.setLeftBoundaryTemp if self.params.boundaryLeft in ["Planckian", "Infinite", "Reflective"] else self.params.boundaryLeft
        T1 = self.params.setRightBoundaryTemp if self.params.boundaryRight in ["Planckian", "Infinite", "Reflective"] else self.params.boundaryRight
        planckLeft = np.broadcast_to(self.planckBarInit(T0), (self.params.freqNum, self.sn))  # shape: (freqNum, sn)
        planckRight = np.broadcast_to(self.planckBarInit(T1), (self.params.freqNum, self.sn))  # shape: (freqNum, sn)
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
        self.sigmaStarVar = self.sigmaStar(self.grid.T_next)  # Update sigma* for the time step
        self.movingMeshConst = self.movingMeshConstant()  # Update moving mesh constant for the time step

    def sigmaBar(self, T):     # Placeholder for group-averaged opacity, currently unnecessary since we are using a constant opacity
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T)
        sbar = self.simpson(integrand, lo, hi)
        return sbar
    
    def psiBar(self):       # placeholder, currently unnecessary due to init
        return

    def polyPredict(self):
        t2 = self.grid.temperatureSet[:, self.grid.timeStep - 2]
        t1 = self.grid.temperatureSet[:, self.grid.timeStep - 1]
        t = self.grid.temperatureSet[:, self.grid.timeStep]
        T_guess = 3*t - 3*t1 + t2
        return T_guess

    def materialEquation(self):
            # Outer constant calc
            dt = self.grid.dt[self.grid.timeStep]
            f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
            
            # Lagged temperature and phi
            if self.grid.timeStep < 3:      
                T_iterative = self.grid.T_next
            elif self.params.extrapolateTemp == True: 
                T_iterative = self.polyPredict()
            else:
                T_iterative = self.grid.T_next
            T = self.grid.temperatureSet[:, self.grid.timeStep]
            phi = self.getPhi()  # Compute scalar flux by integrating over angles
            bbar = self.planckBar(T_iterative)  # Get the group-averaged Planckian for the material 
            sa = self.material.sigma_a(self.grid.freqGroups, T_iterative)
            
            # Calculation of next temperature
            T_offset = self.params.temperatureLearningRate*f * np.sum((sa * phi - 4*np.pi*sa * bbar), axis=0)  # Limit the temperature change to avoid instability
            T_next = np.clip(T + T_offset, a_min=1e-4, a_max=1)  # Update temperature using the material energy equation)
            self.T_next = T_next.copy()
            self.grid.T_next = T_next.copy()

    # This function updates Q*
    def rhsUpdate(self):
        T = self.grid.T_next
        bbarNext = self.planckBar(T)  # Get the group-averaged Planckian for the next temperature
        bbarNext = np.broadcast_to(bbarNext[:, None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(self.material.sigma_a(self.grid.freqGroups, T)[:,None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        rhs = sa * bbarNext + self.timeTerm * self.psi_old
        self.grid.rhs = rhs.copy()

    def movingMeshConstant(self):   # This function calculates the C hat constant for the moving mesh equations
        T = self.grid.T_next
        Tminus = self.grid.temperatureSet[:, self.grid.timeStep] if self.grid.timeStep > 0 else T
        dt = self.grid.dt[self.grid.timeStep]
        dTdt = (T - Tminus) / dt
        dTdx = np.gradient(T, self.grid.dx)
        constSet = np.zeros((self.sn, self.params.nBins))
        for m in range(self.sn):
            constOut = 1/T * (1 / self.const.c * dTdt +  self.mu[m] * dTdx)
            constSet[m] = constOut
        return constSet


    # Define modified opacity
    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGrid, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity
    
    # Here begins helper functions for the moving mesh radiation sweep


    # Helper function to set boundary values based on the side and boundary condition type
    def setBoundaryValues(self, f, m, c, newFull, time, side):
        reflected_m = self.reflMatrix[m]
        
        # 1. Dynamically set spatial side configurations to eliminate copy-paste bugs
        if side == "left":
            bc_type = self.params.boundaryLeft
            # if bc_type != "Reflective" or "Vacuum":
            #     phi_source = self.boundaryCondition("left", time)
            spatial_idx = 0
        elif side == "right":
            bc_type = self.params.boundaryRight
            # if bc_type != "Reflective" or "Vacuum":
            #     phi_source = self.boundaryCondition("right", time)
            spatial_idx = -1
        else:
            raise ValueError("Side must be 'left' or 'right'")

        # 2. Handle Reflective Boundary Condition
        if bc_type == "Reflective":
            # Spatial flux comes from the reflected angle at the current frequency
            bVal = self.fullTens[f, reflected_m, spatial_idx]
        
        if bc_type == "Vacuum":
            bVal = 0.0

        if bc_type == "Planckian":
            phi_source = self.boundaryCondition(side, time)
            bVal = phi_source[f, m]

        if bc_type == "Delta":
            if side == "left":
                T0 = self.params.setLeftBoundaryTemp
            elif side == "right":
                T0 = self.params.setRightBoundaryTemp
            deltaSet = np.zeros((self.params.freqNum, self.sn))
            totalFlux = np.sum(self.grid.du * self.simpson(lambda nu: self.planck(nu, T0), self.grid.freqGrid[:-1], self.grid.freqGrid[1:]))
            deltaSet[self.params.freqNum//4, :] = totalFlux / self.grid.du[self.params.freqNum//4]      # Delta function at the middle frequency group
            bVal = deltaSet[f, m]


        # # 3. Handle Prescribed Source / Inflow Boundary Condition
        # elif bc_type in ["Inflow", "Prescribed"]:
        #     # Spatial flux comes from the external profile
        #     bVal = phi_source[f, m]
                
        return bVal
    
    def setGroupBoundaryValues(self, f, m, c, newFull, time):
        bValGroup = 0.0
        return bValGroup
    
    def freqIndex(self, f, c, sweepDirection):
        if sweepDirection == "forward":
            if c > 0:
                return f - 1
            else:
                return f + 1
        elif sweepDirection == "backward":
            if c > 0:
                return f + 1
            else:
                return f - 1
        else:
            raise ValueError("sweepDirection must be 'forward' or 'backward'")

    
    def pickSweep(self, m, workingSet, newWorkingSet, newFull):
        c = self.movingMeshConst[m, :]  # shape: (nBins,)

        if self.mu[m] >= 0:
            for i in range(self.params.nBins):
                if c[i] > 0:
                    spaceIndex = i - 1
                    sweepDirection = "forward"
                if c[i] <= 0:
                    spaceIndex = i - 1
                    sweepDirection = "backward"

                newWorkingSet = self.radiationBase(c, m, i, sweepDirection, workingSet, newWorkingSet, spaceIndex, newFull)
        elif self.mu[m] < 0:
            for i in range(self.params.nBins-1, -1, -1):
                if c[i] > 0:
                    spaceIndex = i + 1
                    sweepDirection = "forward"
                if c[i] <= 0:
                    spaceIndex = i + 1
                    sweepDirection = "backward"
                newWorkingSet = self.radiationBase(c, m, i, sweepDirection, workingSet, newWorkingSet, spaceIndex, newFull)

        return newWorkingSet
    
    def radiationBase(self, c, m, i, sweepDirection, workingSet, newWorkingSet, spaceIndex, newFull):
        rhs = self.grid.rhs[:, m, i]
        sig_t = self.sigmaStarVar[:, i]

        if sweepDirection == "forward":
            if i == 0 and self.mu[m] >= 0:
                f = 0
                freqIndex = self.freqIndex(f, c[i], sweepDirection)
                bValGroup = self.setGroupBoundaryValues(freqIndex, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep])
                bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "left")
                newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal + abs(c[i]) * self.grid.freqGrid[freqIndex] * bValGroup) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] + abs(c[i]) * self.grid.freqGroups[f])
                for f in range(1, self.freq):
                    freqIndex = self.freqIndex(f, c[i], sweepDirection)
                    bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "left")
                    newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal + abs(c[i]) * self.grid.freqGrid[freqIndex] * workingSet[freqIndex, i]) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] + abs(c[i]) * self.grid.freqGroups[f])
            elif i == self.params.nBins-1 and self.mu[m] < 0:
                f = 0
                freqIndex = self.freqIndex(f, c[i], sweepDirection)
                bValGroup = self.setGroupBoundaryValues(freqIndex, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep])
                bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "right")
                newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal + abs(c[i]) * self.grid.freqGrid[freqIndex] * bValGroup) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] + abs(c[i]) * self.grid.freqGroups[f])
                for f in range(1, self.freq):
                    freqIndex = self.freqIndex(f, c[i], sweepDirection)
                    bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "right")
                    newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal + abs(c[i]) * self.grid.freqGrid[freqIndex] * workingSet[freqIndex, i]) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] + abs(c[i]) * self.grid.freqGroups[f])
            else:
                f = 0
                freqIndex = self.freqIndex(f, c[i], sweepDirection)
                bValGroup = self.setGroupBoundaryValues(freqIndex, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep])
                newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * newWorkingSet[f, spaceIndex] + abs(c[i]) * self.grid.freqGrid[freqIndex] * bValGroup) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] + abs(c[i]) * self.grid.freqGroups[f])
                for f in range(1, self.freq):
                    freqIndex = self.freqIndex(f, c[i], sweepDirection)
                    newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * newWorkingSet[f, spaceIndex] + abs(c[i]) * self.grid.freqGrid[freqIndex] * newWorkingSet[freqIndex, i]) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] + abs(c[i]) * self.grid.freqGroups[f])
        
        if sweepDirection == "backward":
            if i == 0 and self.mu[m] >= 0:
                f = self.freq - 1
                freqIndex = self.freqIndex(f, c[i], sweepDirection)
                bValGroup = self.setGroupBoundaryValues(freqIndex, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep])
                bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "left")
                newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal - abs(c[i]) * self.grid.freqGrid[freqIndex] * bValGroup) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] - abs(c[i]) * self.grid.freqGroups[f])
                for f in range(self.freq - 2, -1, -1):
                    freqIndex = self.freqIndex(f, c[i], sweepDirection)
                    bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "left")
                    newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal - abs(c[i]) * self.grid.freqGrid[freqIndex] * workingSet[freqIndex, i]) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] - abs(c[i]) * self.grid.freqGroups[f])
            elif i == self.params.nBins-1 and self.mu[m] < 0:
                f = self.freq - 1
                freqIndex = self.freqIndex(f, c[i], sweepDirection)
                bValGroup = self.setGroupBoundaryValues(freqIndex, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep])
                bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "right")
                newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal - abs(c[i]) * self.grid.freqGrid[freqIndex] * bValGroup) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] - abs(c[i]) * self.grid.freqGroups[f])
                for f in range(self.freq - 2, -1, -1):
                    freqIndex = self.freqIndex(f, c[i], sweepDirection)
                    bVal = self.setBoundaryValues(f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], "right")
                    newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * bVal - abs(c[i]) * self.grid.freqGrid[freqIndex] * workingSet[freqIndex, i]) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] - abs(c[i]) * self.grid.freqGroups[f])
            else:
                f = self.freq - 1
                freqIndex = self.freqIndex(f, c[i], sweepDirection)
                bValGroup = self.setGroupBoundaryValues(freqIndex, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep])
                newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * newWorkingSet[f, spaceIndex] - abs(c[i]) * self.grid.freqGrid[freqIndex] * bValGroup) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] - abs(c[i]) * self.grid.freqGroups[f])
                for f in range(self.freq - 2, -1, -1):
                    freqIndex = self.freqIndex(f, c[i], sweepDirection)
                    newWorkingSet[f, i] = (rhs[f] + (abs(self.mu[m]) / self.grid.dx) * newWorkingSet[f, spaceIndex] - abs(c[i]) * self.grid.freqGrid[freqIndex] * newWorkingSet[freqIndex, i]) / (abs(self.mu[m]) / self.grid.dx + sig_t[f] - abs(c[i]) * self.grid.freqGroups[f])
        return newWorkingSet
    

    def radiationSweep(self):
            # Initialize time set assets
            self.startTimeStep()

            # Set up Boundary conditions at the time step (allows for variable boundary conditions)
            time = self.grid.timeSet[self.grid.timeStep]

            # Initialize the next tensor
            newFull = np.zeros_like(self.fullTens)

            # Loop through Angles
            for m in range(self.sn):
                workingSet = self.fullTens[:, m, :].copy()  # shape: (freqNum, nBins)
                newWorkingSet = newFull[:, m, :].copy()  # shape: (freqNum, nBins)
                newWorkingSet = self.pickSweep(m, workingSet, newWorkingSet, newFull)
                newFull[:, m, :] = newWorkingSet

            # Set fullTensor to the updated Tensor
            self.grid.fullTensor = newFull.copy()



class Logic:
    def __init__(self, params, grid, material, constants):
        self.grid = grid
        self.constants = constants
        self.params = params
        self.material = material
        
    
        if not self.params.materialCoupled:
            self.equations = Equations(params, grid, material, constants)

        elif self.params.materialCoupled and not self.params.movingCoordinates:
            self.equations = CoupledEquations(params, grid, material, constants)
            self.materialEquation = self.equations.materialEquation  # Expose material equation for coupled problems
        
        elif self.params.materialCoupled and self.params.movingCoordinates:
            self.equations = MovingMeshEquations(params, grid, material, constants)
            self.materialEquation = self.equations.materialEquation  # Expose material equation for coupled problems with moving mesh

    def applyInitialConditions(self):
        self.equations.applyInitialConditions()

    def radiationSweep(self):
        self.equations.radiationSweep()
    
    def rhsUpdate(self):
        self.equations.rhsUpdate()

    def getPhi(self):
        return self.equations.getPhi()

    def simpson(self, integrand, lo, hi):
        return self.equations.simpson(integrand, lo, hi)

