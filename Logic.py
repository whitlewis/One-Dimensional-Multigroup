import numpy as np


# Non coupled equations
class Equations:

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
        self.timeAbsorption = 0.0
        self.mu = self.grid.muSet


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
        self.params = params
        self.grid = grid
        self.material = material
        self.const = constants
        self.fullTens = grid.fullTensor.copy()  # shape: (freqNum, sn, nBins)
        self.freq = params.freqNum
        self.sn = params.sn
        self.dx = grid.dx
        self.time_step = None
        self.timeTerm = 0.0
        self.sigma_a = self.material.sigma_a()

    # Simpson for Plack integration over group
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
        T0 = self.params.initialTemperature
        planck = self.groupPlanck(T0) 
        self.grid.fullTensor[:] = planck[:, None, None]
        self.grid.updateFullTensor(self.grid.fullTensor)

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

    # Definition of the coupled material equation
    def materialEquation(self):

        dt = self.grid.dt[self.grid.timeStep]
        f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
        T = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature in all x cells (120,1)
        phi = np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=0)  # Compute scalar flux by integrating over angles
        bbar = self.groupPlanck(T)  # Get the group-averaged Planck source

        T_next = T + f * np.sum((self.material.sigma_a(self.grid.freqGrid, T) * phi - self.material.sigma_a(self.grid.freqGrid, T) * bbar), axis=0)  # Update temperature using the material energy equation)
        self.grid.temperatureSet[:, self.grid.timeStep] = T_next  # Update the temperature set for the current time step
        rhs = self.material.sigma_a(self.grid.freqGroups, T_next) * bbar - self.material.sigma_a(self.grid.freqGroups, T_next) * phi  # Right-hand side of the transport equation
        return T_next, rhs
    
    # Define modified opacity
    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGrid, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity

    def radiationSweep(self):
        # Initialize time set assets
        self.startTimeStep()


        mu = self.grid.muSet
        dx = self.grid.dx
        newfull = np.zeros_like(self.grid.fullTensor)

        # Compute RHS once per sweep
        T_next, rhs = self.materialEquation(self.grid.fullTensor)
        rhs += self.timeTerm * self.psi_old
        # print("RHS shape:", rhs.shape)  # Debug shape of RHS
        # print("T_next shape:", T_next.shape)  # Debug shape of T_next
        # These are now (nFreq, nMu)
        phiBl = self.groupPlanck(self.params.sourceTemp) # Placeholder for boundary layer contribution
        phiBr = np.zeros(self.grid.fullTensor[:,:,0].shape)  # Placeholder for boundary reflection contribution
        # print(phiBl.shape, phiBr.shape, rhs.shape)  # Debug shapes

        # We only loop over angles (m). Frequency (f) is handled by the ":"
        for m, mu_val in enumerate(mu):
            if mu_val > 0:
                # --- Forward sweep ---
                # Boundary cell i=0 (Vectorized over all f)
                newfull[:, m, 0] = (rhs[m, 0] + (mu_val / dx) * phiBl[:, m]) / \
                                (mu_val / dx + self.sigmaStar(T_next[0]))
                
                for i in range(self.params.nBins - 1):
                    # Interior cells (Vectorized over all f)
                    newfull[:, m, i + 1] = (rhs[m, i + 1] + (mu_val / dx) * self.grid.fullTensor[:, m, i]) / \
                                        (mu_val / dx + self.sigmaStar(T_next[i + 1]))
            else:
                # --- Backward sweep ---
                # Boundary cell i=-1
                # Use mu[m] (a single number) instead of mu (the whole array)
                newfull[:, m, -1] = (rhs[m, -1] + (abs(mu_val) / self.grid.dx) * phiBr[:, m]) / \
                                    (abs(mu_val) / self.grid.dx + self.sigmaStar(T_next[-1]))
                
                for i in range(self.params.nBins - 1, 0, -1):
                    newfull[:, m, i - 1] = (rhs[m, i - 1] + (mu_val / dx) * self.grid.fullTensor[:, m, i]) / \
                                        (mu_val / dx + self.sigmaStar(T_next[i - 1]))
        self.grid.fullTensor = newfull.copy()
        self.grid.temperatureSet[:, self.grid.timeStep] = T_next  # Update the temperature set for the current time step


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

