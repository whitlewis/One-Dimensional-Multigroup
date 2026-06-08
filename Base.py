import numpy as np
import numpy.polynomial.legendre as leggauss
import time as machineTime

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
        self.fullTensorPhi = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, nBins)
        self.fullTensorPhiTime = np.zeros((parameters.nSteps+1,) + self.fullTensorPhi.shape)  # shape: (nSteps+1, freqNum, nBins)

        # Helper variables for calculations
        self.timeStep = 0   # time step counter
        self.rhsfull = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, sn, nBins)
        self.psiOld = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # shape: (freqNum, sn, nBins)




class Base:
    def __init__(self, grid, problem, params, constants):
        self.grid = grid
        self.problem = problem
        self.constants = constants
        self.params = params
        self.index = 0  # Initialize index for time stepping
        self.getPhi = self.problem.equations.getPhi  # Initialize getPhi method from the equations object for use in storing scalar flux each step
    
    def converge(self):
        for it in range(self.params.maxIters):

            # update Temperature and get Q* if material coupled
            if self.params.materialCoupled:
                self.problem.equations.rhsUpdate()
            # Perform the radiation sweep to get the new solution
            self.problem.equations.radiationSweep()
            if self.params.checkEnergy and self.index % self.params.energyCheckFreq == 0 and it==0:  # Check energy conservation every 50 time steps
                energyConserved = self.checkEnergyConservation()
            if self.params.materialCoupled:
                self.problem.equations.materialEquation() # update material temperature after radiation sweep

            err = np.max(np.abs((self.grid.fullTensor - self.fullTensOld)))    # directly compare the full values for convergence (Space, angle, and frequency group convergence)
            if np.isnan(err).any() or np.isinf(err).any():
                name = "Convergence Check"
                print("\n⚠️ INVALID RESULT DETECTED")
                print("Operation:", name)
                print("a =",self.grid.fullTensor)
                print("b =", self.fullTensOld)
                print("result =", err)
            if err < self.params.tol:
                if it > 40: print(f"Converged in {it} iterations")
                break
    
    def updateAll(self, index):
        # Updates grid object with new solutions each step
        self.grid.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
        self.grid.fullTensorTime[index] = self.grid.fullTensor.copy()  # Store the solution for this time step
        self.grid.fullTensorPhiTime[index] = self.getPhi().copy()  # Store scalar flux for this time step
        self.grid.timeStep += 1  # Increment time step counter
        if self.params.materialCoupled:                      # update temperature for next step
            self.grid.temperatureSet[:, self.grid.timeStep] = self.problem.equations.equations.T_next

    def setEnergy(self):
        if self.params.materialCoupled:
            Tmat = self.grid.temperatureSet[:, self.grid.timeStep]  
            Emat = self.problem.material.C_v(Tmat) * Tmat  
            Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep], axis=0) / self.constants.c  
            totalEnergy = np.sum((Emat + Erad) * self.grid.dx) 
            self.params.totalEnergy = totalEnergy  
            print("Initial energy set:")
            print(f"Material energy: {np.sum(Emat * self.grid.dx):.4e}, Radiation energy: {np.sum(Erad * self.grid.dx):.4e}")
            print(f"Total energy: {totalEnergy:.4e}") 
        else:
            Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep], axis=0) / self.constants.c  
            totalEnergy = np.sum(Erad * self.grid.dx)  
            self.params.totalEnergy = totalEnergy  
            print("Initial energy set:")
            print(f"Radiation energy: {totalEnergy:.4e}")
            print(f"Total energy: {totalEnergy:.4e}")
        
        
    def checkEnergyConservation(self):
        if not self.params.materialCoupled:
            Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep], axis=0) / self.constants.c  
            totalEnergy = np.sum(Erad * self.grid.dx)  
            time = self.grid.timeSet[self.grid.timeStep]
            dt = self.grid.dt[self.grid.timeStep]
            sourceEnergy = np.sum(self.problem.material.source(self.grid.fullTensor) * self.grid.dx) * dt 
            print(f"Energy from sources at time {time:.2e}: {sourceEnergy:.4e}")
            self.params.totalEnergy += sourceEnergy  
            print(f"Total energy after accounting for sources: {self.params.totalEnergy:.4e}")
            diffEnergy = np.abs(totalEnergy - self.params.totalEnergy)  
            if diffEnergy > self.params.energyTol:
                print(f"⚠️ Energy conservation check failed at time step {self.grid.timeStep} (time={self.grid.timeSet[self.grid.timeStep]:.2e})")
                print(f"Max energy difference: {diffEnergy:.2e}")
                print(f"Radiation energy: {totalEnergy:.4e}")
                print(f"Total energy: {totalEnergy:.4e}")
                return False
        elif self.params.materialCoupled:
            Tmat = self.grid.temperatureSet[:, self.grid.timeStep]  
            Emat = self.problem.material.C_v(Tmat) * Tmat  
            Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep], axis=0) / self.constants.c  
            Trad = (Erad / self.constants.a)**0.25  
            totalEnergy = np.sum((Emat + Erad) * self.grid.dx)  
            diffEnergy = np.abs(totalEnergy - self.params.totalEnergy)  
            if diffEnergy > self.params.energyTol:
                print(f"⚠️ Energy conservation check failed at time step {self.grid.timeStep} (time={self.grid.timeSet[self.grid.timeStep]:.2e})")
                print(f"Max energy difference: {diffEnergy:.2e}")
                print(f"Material energy: {np.sum(Emat * self.grid.dx):.4e}, Radiation energy: {np.sum(Erad * self.grid.dx):.4e}")
                print(f'material temperature: {Tmat[20]:.4e}, radiation temperature: {Trad[20]:.4e}')
                print(f"Total energy: {totalEnergy:.4e}")
                return False

        print(f"Energy conservation check passed at time step {self.grid.timeStep-1} (time={self.grid.timeSet[self.grid.timeStep]:.2e})")
        print(f"Max energy difference: {diffEnergy:.8e}")
        if self.params.materialCoupled:
            print(f"Material energy: {np.sum(Emat * self.grid.dx):.4e}, Radiation energy: {np.sum(Erad * self.grid.dx):.4e}")
        else:
            print(f"Radiation energy: {totalEnergy:.4e}")
        print(f"Total energy: {totalEnergy:.4e}")
        return True
    
    def solve(self):
        tick = machineTime.perf_counter()
        self.grid.fullTensorTime[0] = self.grid.fullTensor.copy()    # Initialize 0'th step (Thanks to Johannes for this fix)
        self.grid.fullTensorPhiTime[0] = self.getPhi()
        self.grid.fullTensOld = self.grid.fullTensor.copy()  # Initialize old solution for time-stepping
        self.setEnergy()  # Initial energy check

        print("Starting Solve...") 
        for index, time in enumerate(self.grid.timeSet[:-1]):
            self.index = index  # Update index for this time step
            self.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
            self.converge()  # Perform the radiation sweep to get the new solution
            self.updateAll(index)  # Store the solution and increment time step counter
            if index % 200 == 0:  
                print(f"Completed time step {time:.2e}")
        tock = machineTime.perf_counter()
        print(f"Solve completed in {tock - tick:.2f} seconds.")
        
        return self.grid.fullTensorTime, self.grid  # Return the full time-dependent solution tensor
    


    
