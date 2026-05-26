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
        if parameters.freqNum > 1:
            self.freqGrid = np.logspace(-12, np.log10(parameters.maxFreq), parameters.freqNum + 1)  # logarithmic spacing over group boundaries
        else:
            self.freqGrid = np.array([1e-3, parameters.maxFreq])  # single frequency case
        self.freqGroups = 0.5 * (self.freqGrid[:-1] + self.freqGrid[1:])
        self.fullTensor = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # (nfreq, nMu, nBins)
        self.muSet, self.w = np.polynomial.legendre.leggauss(parameters.sn)  # Gauss-Legendre quadrature points and weights for angular discretization
        self.w /= 2.0  # Normalize weights to sum to 1

        if parameters.timeScale == "log":
            self.timeSet = np.logspace(-12, np.log10(parameters.timeMax), parameters.nSteps+1)  # logarithmic time steps (could be linear)
        
        if parameters.timeScale == 'linear':
            self.timeSet = np.linspace(0, parameters.timeMax, parameters.nSteps+1)  # linear time steps
        self.dt = np.diff(self.timeSet)  # time step sizes

        self.timeStep = 0
        self.temperatureSet = np.ones((parameters.nBins, parameters.nSteps+1))*parameters.initialTemperature  # Initialize temperature set for all time steps
        self.fullTensorTime = np.zeros((parameters.nSteps+1,) + self.fullTensor.shape)  # shape: (nSteps+1, freqNum, nMu, nBins)
        self.fullTensorPhi = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, nMu, nBins)
        self.fullTensorPhiTime = np.zeros((parameters.nSteps+1,) + self.fullTensorPhi.shape)  # shape: (nSteps+1, freqNum, nBins)
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
            self.problem.equations.radiationSweep()  # Perform the radiation sweep to get the new solution
            err = np.max(np.abs((self.grid.fullTensor - self.fullTensOld)))    # directly compare the full values for convergence
            if np.isnan(err).any() or np.isinf(err).any():
                name = "Convergence Check"
                print("\n⚠️ INVALID RESULT DETECTED")
                print("Operation:", name)
                print()
                print("a =",self.grid.fullTensor)
                print("b =", self.fullTensOld)
                print("result =", err)
                assert 0
            if err < self.params.tol:
                if it > 10: print(f"Converged in {it} iterations")
                break
            self.fullTensOld = self.grid.fullTensor.copy()

    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi
    
    def updateAll(self, index):
        self.grid.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
        self.grid.fullTensorTime[index] = self.grid.fullTensor.copy()  # Store the solution for this time step
        self.grid.fullTensorPhiTime[index] = self.getPhi().copy()  # Store scalar flux for this time step
        self.grid.timeStep += 1  # Increment time step counter
    
    def solve(self):
        self.grid.fullTensorTime[0] = self.grid.fullTensor.copy()    # Initialize 0'th step (Thanks to Johannes for this fix)
        self.grid.fullTensorPhiTime[0] = self.getPhi()
        self.grid.fullTensOld = self.grid.fullTensor.copy()  # Initialize old solution for time-stepping
        
        print("Starting Solve...") 
        for index, time in enumerate(self.grid.timeSet[:-1]):
            self.fullTensOld = self.grid.fullTensor.copy()  # Update old solution for time-stepping
            self.converge()  # Perform the radiation sweep to get the new solution
            self.updateAll(index)  # Store the solution and increment time step counter
            if index % 200 == 0:  
                print(f"Completed time step {time:.2e}")
        print("Solve completed.")
        
        return self.grid.fullTensorTime, self.grid  # Return the full time-dependent solution tensor
    


    
