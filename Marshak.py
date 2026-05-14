import numpy as np
import numpy.polynomial.legendre as leggauss

class Parameters:
    def __init__(self, maxIters=1000, tol=1e-10, nSteps=1000):
        self.maxIters = maxIters
        self.tol = tol
        self.nSteps = nSteps
        self.nBins = 1
        self.xMin = 0
        self.xMax = 1
        self.sn = 2
        self.freqNum = 10
        self.timeMax = 20
        self.maxFreq = 150
        self.initialTemperature = 0.0
        self.sourceTemp = 1.0

class Material:
    def __init__(self, params, grid):
        self.params = params
        self.grid = grid
    
    def sigma_a(self, freq, T): # Placeholder constant opacity
        return 100
    
    def C_v(self, T):  # Placeholder constant heat capacity
        return 1.0
    
    def addSource(self):
        # Add a source term at the left boundary (x=0) for all frequencies and angles
        source = np.zeros((self.grid.freqNum, self.grid.sn))
        source[:, :] = self.params.sourceTemp  # Set the source temperature for all frequencies and angles
        return source


class Equations:

    def __init__(self, params, grid, material, constants):
        self.params = params
        self.grid = grid
        self.material = material
        self.const = constants

    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
        return out

    def planck(self, nu, T):  # Planck function (not group integrated or weighted)
        denom = np.expm1(nu/T)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * nu**3 / denom

    def groupPlanck(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar

    def initSpectra(self):
        T0 = self.params.initialTemperature
        planck = self.groupPlanck(T0) 
        self.grid.fullTensor[:] = planck[:, None, None]
        self.grid.updateFullTensor(self.grid.fullTensor)


    def materialEquation(self, fullTensor):
        dt = self.grid.dt[self.grid.timeStep]
        f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
        T = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature in all x cells (120,1)
        phi = np.sum(self.grid.w[:, None] * fullTensor, axis=0)  # Compute scalar flux by integrating over angles
        bbar = self.groupPlanck(T)  # Get the group-averaged Planck source

        T_next = T + f * np.sum((self.material.sigma_a(self.grid.freqGrid, T) * phi - self.material.sigma_a(self.grid.freqGrid, T) * bbar), axis=0)  # Update temperature using the material energy equation)
        self.grid.temperatureSet[:, self.grid.timeStep] = T_next  # Update the temperature set for the current time step
        rhs = self.material.sigma_a(self.grid.freqGroups, T_next) * bbar - self.material.sigma_a(self.grid.freqGroups, T_next) * phi  # Right-hand side of the transport equation
        return T_next, rhs

    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGrid, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity


    def radiationSweep(self):
        mu = self.grid.muSet
        dx = self.grid.dx
        newfull = np.zeros_like(self.grid.fullTensor)

        # Compute RHS once per sweep (crucial for performance)
        T_next, rhs = self.materialEquation(self.grid.fullTensor)
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
        return self.grid, newfull, T_next


class Marshak:
    def __init__(self, grid, constants):
        self.parameters = Parameters()
        self.material = Material(self.parameters, grid)
        self.equations = Equations(self.parameters, grid, self.material, constants)

        