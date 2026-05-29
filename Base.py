import numpy as np
import numpy.polynomial.legendre as leggauss

class Constants:
    # all physical constants
    c = 1.0  
    a = 0.01374
    h = 1.0

class Grid:
    def __init__(self, parameters, Constants=Constants()):

        # Space grid
        self.dx = (parameters.xMax - parameters.xMin) / parameters.nBins
        self.spaceGrid = np.linspace(parameters.xMin, parameters.xMax, parameters.nBins + 1)  # cell edges
        self.spaceMid = 0.5 * (self.spaceGrid[:-1] + self.spaceGrid[1:])  # cell centers

        # Frequency grid
        if parameters.freqNum > 2:
            self.freqGrid = np.append(np.logspace(np.log10(parameters.minFreq), np.log10(parameters.maxFreq), parameters.freqNum), parameters.infFreq)  # logarithmic spacing over group boundaries
        elif parameters.freqNum == 2:
            self.freqGrid = np.array([1e-3, parameters.maxFreq, parameters.infFreq])  # two groups: one for low frequencies and one for high frequencies
        else:
            self.freqGrid = np.array([1e-3, parameters.maxFreq])  # single frequency case

        # Gives midpoints of frequency groups no matter number
        self.freqGroups = 0.5 * (self.freqGrid[:-1] + self.freqGrid[1:])


        # Angular discretization
        self.muSet, self.w = np.polynomial.legendre.leggauss(parameters.sn)  # Gauss-Legendre quadrature points and weights for angular discretization
        self.w /= 2.0  # Normalize weights to sum to 1

        # Time discretization (log or linear spaced)
        if parameters.timeScale == "log":
            self.timeSet = np.logspace(-12, np.log10(parameters.timeMax), parameters.nSteps+1)  # logarithmic time steps (could be linear)
        
        if parameters.timeScale == 'linear':
            self.timeSet = np.linspace(0, parameters.timeMax, parameters.nSteps+1)  # linear time steps
        self.dt = np.diff(self.timeSet)  # time step sizes

        # Individual time step frameworks
        self.fullTensor = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # (nfreq, nMu, nBins)
        self.temperatureSet = np.zeros((parameters.nBins, parameters.nSteps+1)) # Initialize temperature set for all time steps
        self.temperatureSet[:, 0] = parameters.initialTemperature  # Set initial temperature distribution at time step 0
        self.T_next = self.temperatureSet[:, 0].copy()  # Initialize T_next for the first step

        # Time-dependent tensors
        self.fullTensorTime = np.zeros((parameters.nSteps+1,) + self.fullTensor.shape)  # shape: (nSteps+1, freqNum, nMu, nBins)
        self.fullTensorPhi = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, nMu, nBins)
        self.fullTensorPhiTime = np.zeros((parameters.nSteps+1,) + self.fullTensorPhi.shape)  # shape: (nSteps+1, freqNum, nBins)

        # Helper variables for calculations
        self.timeStep = 0   # time step counter
        self.rhsfull = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, sn, nBins)
        self.psiOld = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # shape: (freqNum, sn, nBins)

    def updateFullTensor(self, newFull):
        self.fullTensor = newFull.copy()



class Base:
    def __init__(self, grid, problem, params, constants):
        self.grid = grid
        self.problem = problem
        self.constants = constants
        self.params = params
    
    def converge(self):
        for it in range(self.params.maxIters):
            # update Temperature and get Q*
            self.problem.equations.materialEquation()

            # Perform the radiation sweep to get the new solution
            self.problem.equations.radiationSweep()  

            err = np.max(np.abs((self.grid.fullTensor - self.fullTensOld)))    # directly compare the full values for convergence (Space, angle, and frequency group convergence)
            if np.isnan(err).any() or np.isinf(err).any():
                name = "Convergence Check"
                print("\n⚠️ INVALID RESULT DETECTED")
                print("Operation:", name)
                print()
                print("a =",self.grid.fullTensor)
                print("b =", self.fullTensOld)
                print("result =", err)
            if err < self.params.tol:
                if it > 40: print(f"Converged in {it} iterations")
                break
            self.fullTensOld = self.grid.fullTensor.copy()
            

    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi
    
    def updateAll(self, index):
        # Updates grid object with new solutions each step
        self.grid.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
        self.grid.fullTensorTime[index] = self.grid.fullTensor.copy()  # Store the solution for this time step
        self.grid.fullTensorPhiTime[index] = self.getPhi().copy()  # Store scalar flux for this time step
        self.grid.timeStep += 1  # Increment time step counter
        if self.params.materialCoupled:                      # update temperature for next step
            self.grid.temperatureSet[:, self.grid.timeStep] = self.problem.equations.equations.T_next

    def setEnergy(self):
        Tmat = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature distribution
        Emat = self.problem.material.C_v(Tmat) * Tmat  # Energy density of the material
        Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep] * self.grid.freqGroups[:, None], axis=0)  # Energy density of the radiation
        totalEnergy = Emat + Erad  # Total energy density
        self.grid.totalEnergy = totalEnergy  # Store total energy for comparison
        print("Initial energy set:")
        print(f"Material energy: {np.sum(Emat):.4e}, Radiation energy: {np.sum(Erad):.4e}")
        print(f"Total energy: {np.sum(totalEnergy):.4e}")    
    
    
    def checkEnergyConservation(self):
        Tmat = self.grid.temperatureSet[:, self.grid.timeStep]  # Current temperature distribution
        Emat = self.problem.material.C_v(Tmat) * Tmat  # Energy density of the material
        Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep] * self.grid.freqGroups[:, None], axis=0)  # Energy density of the radiation
        totalEnergy = Emat + Erad  # Total energy density
        diffEnergy = np.abs(totalEnergy - self.grid.totalEnergy)  # Change in total energy from previous time step
        if np.any(diffEnergy > self.params.energyTol):
            print(f"⚠️ Energy conservation check failed at time step {self.grid.timeStep} (time={self.grid.timeSet[self.grid.timeStep]:.2e})")
            print(f"Max energy difference: {np.max(diffEnergy):.2e}")
            print(f"Material energy: {np.sum(Emat):.4e}, Radiation energy: {np.sum(Erad):.4e}")
            print(f"Total energy: {np.sum(totalEnergy):.4e}")
            return False

        print(f"Energy conservation check passed at time step {self.grid.timeStep} (time={self.grid.timeSet[self.grid.timeStep]:.2e})")
        print(f"Max energy difference: {np.max(diffEnergy):.8e}")
        print(f"Material energy: {np.sum(Emat):.4e}, Radiation energy: {np.sum(Erad):.4e}")
        print(f"Total energy: {np.sum(totalEnergy):.4e}")
        return True
    
    def solve(self):
        self.grid.fullTensorTime[0] = self.grid.fullTensor.copy()    # Initialize 0'th step (Thanks to Johannes for this fix)
        self.grid.fullTensorPhiTime[0] = self.getPhi()
        self.grid.fullTensOld = self.grid.fullTensor.copy()  # Initialize old solution for time-stepping
        self.setEnergy()  # Initial energy check

        print("Starting Solve...") 
        for index, time in enumerate(self.grid.timeSet[:-1]):
            self.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
            self.converge()  # Perform the radiation sweep to get the new solution
            self.updateAll(index)  # Store the solution and increment time step counter
            if self.params.checkEnergy and index % 50 == 0:  # Check energy conservation every 50 time steps
                energyConserved = self.checkEnergyConservation()
                if not energyConserved:
                    print(f"Energy conservation check failed at time step {index} (time={time:.2e})")
            if index % 200 == 0:  
                print(f"Completed time step {time:.2e}")
        print("Solve completed.")
        
        return self.grid.fullTensorTime, self.grid  # Return the full time-dependent solution tensor
    


    
