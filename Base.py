import numpy as np
import numpy.polynomial.legendre as leggauss

class Constants:
    c = 1.0  
    a = 0.01374
    h = 1.0

class Grid:
    def __init__(self, parameters, Constants=Constants()):
        self.dx = (parameters.xMax - parameters.xMin) / parameters.nBins
        self.spaceGrid = np.linspace(parameters.xMin, parameters.xMax, parameters.nBins + 1)  # cell edges
        self.spaceMid = 0.5 * (self.spaceGrid[:-1] + self.spaceGrid[1:])  # cell centers
        self.freqGrid = np.append(np.logspace(-12, np.log10(25), parameters.freqNum - 1), parameters.maxFreq)  # logarithmic spacing
        self.freqGroups = 0.5 * (self.freqGrid[:-1] + self.freqGrid[1:])
        self.fullTensor = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # (nfreq, nMu, nBins)
        self.muSet, self.w = np.polynomial.legendre.leggauss(parameters.sn)  # Gauss-Legendre quadrature points and weights for angular discretization
        self.w /= 2.0  # Normalize weights to sum to 1
        self.timeSet = np.logspace(-12, np.log10(parameters.timeMax), parameters.nSteps+1)  # logarithmic time steps (could be linear)
        self.dt = np.diff(self.timeSet)  # time step sizes
        self.timeStep = 0
        self.temperatureSet = np.ones((parameters.nBins, parameters.nSteps+1))*parameters.initialTemperature  # Initialize temperature set for all time steps
        self.fullTensorTime = np.zeros((parameters.nSteps+1,) + self.fullTensor.shape)  # shape: (nSteps+1, freqNum, nMu, nBins)
        self.fullTensorPhi = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, nMu, nBins)
        self.fullTensorPhiTime = np.zeros((parameters.nSteps+1,) + self.fullTensorPhi.shape)  # shape: (nSteps+1, freqNum, nBins)
        
    
    def updateFullTensor(self, newFull):
        self.fullTensor = newFull.copy()



class Base:
    def __init__(self, grid, problem, params, constants):
        self.grid = grid
        self.problem = problem
        self.constants = constants
        self.params = params
    
    def converge(self):
        self.fullTensOld = self.grid.fullTensor.copy()
        rhsfull = self.problem.material.source(self.fullTensOld)  
        for it in range(self.params.maxIters):
            self.grid, newFull, T_next = self.problem.equations.radiationSweep()  # Perform the radiation sweep to get the new solution
            err = np.max(np.abs((newFull - self.fullTensOld)))    # directly compare the full values for convergence
            if np.isnan(err).any() or np.isinf(err).any():
                name = "Convergence Check"
                print("\n⚠️ INVALID RESULT DETECTED")
                print("Operation:", name)
                print("a =",newFull)
                print("b =", self.fullTensOld)
                print("result =", err)
            if err < self.params.tol:
                print(f"Converged in {it} iterations")
                break
            self.fullTensOld = newFull.copy()

        self.grid.fullTensor = newFull.copy()        
        return self.grid, newFull, T_next

    def getPhi(self, fullTensor):
        # Integrate over angles to get scalar flux
        fullTensorPhi = np.sum(self.grid.w[:, None] * fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi

    
    def solve(self):
        print("Starting Solve...")
        for index, time in enumerate(self.grid.timeSet[:-1]):
            self.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
            # T, rhsfull = self.problem.equations.materialEquation(self.fullTensOld)  # Compute the right-hand side for the current time step
            self.grid, newFull, T_next = self.converge(self.grid, self.problem)  # Perform the radiation sweep to get the new solution
            self.grid.fullTensorTime[index] = newFull.copy()  # Store the solution for this time step
            self.fullTensorPhi = self.getPhi(newFull)  # Compute scalar flux for this time step
            self.grid.fullTensorPhiTime[self.grid.timeStep] = self.fullTensorPhi.copy()  # Store scalar flux for this time step
            self.grid.timeStep += 1  # Increment time step counter
            if index % 200 == 0:  
                print(f"Completed time step {time:.2e}")
        print("Solve completed.")
        
        return self.grid.fullTensorTime, self.grid  # Return the full time-dependent solution tensor
    


    
