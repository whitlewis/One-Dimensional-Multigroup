import numpy as np
import numpy.polynomial.legendre as leggauss
import time as machineTime
import h5py
from datetime import datetime

class Constants:
    # all physical constants
    c = 29.9792458 # speed of light in cm/ns
    a = 0.0137202
    h = 1.0  # Frequency grid in keV

class Grid:
    def __init__(self, parameters, Constants=Constants()):
        self.constants = Constants
        # Space grid
        self.nBins = parameters.nBins
        self.nSteps = parameters.nSteps
        self.dx = (parameters.xMax - parameters.xMin) / parameters.nBins
        self.spaceGrid = np.linspace(parameters.xMin, parameters.xMax, parameters.nBins + 1)  # cell edges
        self.spaceMid = 0.5 * (self.spaceGrid[:-1] + self.spaceGrid[1:])  # cell centers

        # Frequency grid
        if parameters.freqNum > 2:
            if parameters.groupSpace == 'log':
                self.freqGrid = np.append(np.logspace(np.log10(parameters.minFreq), np.log10(parameters.maxFreq), parameters.freqNum), parameters.infFreq)  # logarithmic spacing over group boundaries
            elif parameters.groupSpace == 'linear':
                self.freqGrid = np.append(np.linspace(parameters.minFreq, parameters.maxFreq, parameters.freqNum), parameters.infFreq)
        elif parameters.freqNum == 2:
            self.freqGrid = np.array([1e-3, parameters.maxFreq, parameters.infFreq])  # two groups: one for low frequencies and one for high frequencies
        else:
            self.freqGrid = np.array([1e-3, parameters.maxFreq])  # single frequency case

        # Gives midpoints of frequency groups no matter number
        self.freqGroups = 0.5 * (self.freqGrid[:-1] + self.freqGrid[1:])
        self.du = np.diff(self.freqGrid)  # frequency group widths


        # Angular discretization
        self.muSet, self.w = np.polynomial.legendre.leggauss(parameters.sn)  # Gauss-Legendre quadrature points and weights for angular discretization
        self.w /= 2.0  # Normalize weights to sum to 1

        # Time discretization (log or linear spaced)
        if parameters.logLinTime == "Split":
            stepsLog = round(parameters.stepSplit * self.nSteps)
            stepsLin = self.nSteps - stepsLog
            logSet = np.logspace(-12, parameters.timeSplit, stepsLog, endpoint=False)
            linSet = np.linspace(parameters.timeSplit, parameters.timeMax, stepsLin + 1)
            self.timeSet = np.concatenate(logSet, linSet, axis=0)
        else:
            if parameters.timeScale == "log":
                self.timeSet = np.logspace(-12, np.log10(parameters.timeMax), parameters.nSteps+1)  # logarithmic time steps (could be linear)
            if parameters.timeScale == 'linear':
                self.timeSet = np.linspace(0, parameters.timeMax, parameters.nSteps+1)  # linear time steps
        self.dt = np.diff(self.timeSet)  # time step sizes
        if np.max(self.dt) > 1e-3:
            print(f'Max time step exceeds recommended. Max time step of: {np.max(self.dt)}')

        # Individual time step frameworks
        self.fullTensor = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # (nfreq, nMu, nBins)
        self.fullTensOld = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # (nfreq, nMu, nBins)
        self.temperatureSet = np.zeros((parameters.nBins, parameters.nSteps+1)) # Initialize temperature set for all time steps
        self.temperatureSet[:, 0] = parameters.initialTemperature  # Set initial temperature distribution at time step 0
        self.T_next = self.temperatureSet[:, 0].copy()  # Initialize T_next for the first step

        # Time-dependent tensors
        self.fullTensorTime = np.zeros((parameters.nSteps+1,) + self.fullTensor.shape)  # shape: (nSteps+1, freqNum, nMu, nBins)
        self.fullTensorPhi = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, nBins)
        self.fullTensorPhiTime = np.zeros((parameters.nSteps+1,) + self.fullTensorPhi.shape)  # shape: (nSteps+1, freqNum, nBins)

        # Helper variables for calculations
        self.timeStep = 0   # time step counter
        self.rhs = np.zeros((parameters.freqNum, parameters.nBins))  # shape: (freqNum, sn, nBins)
        self.psiOld = np.zeros((parameters.freqNum, parameters.sn, parameters.nBins))  # shape: (freqNum, sn, nBins)




class Base:
    def __init__(self, grid, problem, params, constants):
        self.grid = grid
        self.problem = problem
        self.constants = constants
        self.params = params
        self.index = 0  # Initialize index for time stepping
        self.getPhi = self.problem.equations.getPhi  # Initialize getPhi method from the equations object for use in storing scalar flux each step
        self.err = 0.0
        self.errorStag = 0.0
        self.errorLag= 0.0

    
    def converge(self):
        for it in range(self.params.maxIters):

            # update Temperature and get Q* if material coupled
            if self.params.materialCoupled:
                self.rhs = self.grid.rhs.copy()
                self.problem.equations.rhsUpdate()
            # Perform the radiation sweep to get the new solution
            self.problem.equations.radiationSweep()
            if self.params.checkEnergy and self.index % self.params.energyCheckFreq == 0 and it==0:  # Check energy conservation every 50 time steps
                self.checkEnergyConservation()
            if self.params.materialCoupled:
                self.T_next = self.grid.T_next.copy()
                self.problem.equations.materialEquation() # update material temperature after radiation sweep
            diff = np.abs((self.grid.fullTensor - self.grid.fullTensOld))
            err = np.max(diff)    # directly compare the full values for convergence (Space, angle, and frequency group convergence)
            if self.params.iterationCheck == True:
                if it % 10 == 0:

                    print(f"Time step {self.index}, Iteration {it}, Error change: {self.err - err:.2e}")
                    print(f'Old error: {self.err}, New error{err}, Mean Error {np.mean(diff)}')
                    self.err = err  # Store the error for external access if needed
                    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

                    val_new = self.grid.fullTensor[max_idx]
                    val_old = self.grid.fullTensOld[max_idx]

                    print(f"Max Error: {err:.6e}")
                    print(f"Occurred at Index: {max_idx}")
                    print(f"  fullTensor: {val_new}")
                    print(f"  fullTensOld: {val_old}")
            if np.isnan(err).any() or np.isinf(err).any():
                name = "Convergence Check"
                print("\n⚠️ INVALID RESULT DETECTED")
                print("Operation:", name)
                print("a =",self.grid.fullTensor)
                print("b =", self.grid.fullTensOld)
                print("result =", err)
            if err < self.params.tol:
                if it > 40: print(f"Converged in {it} iterations")
                break
            if it % 10 == 0:
                self.errorLag = err
            if abs(self.err - err) < 1e-40 and abs(self.errorLag - err) < 1e-40:
                self.errorStag += 1
                if self.errorStag > 4:
                    print(f'Not further trending toward convergence, breaking loop and Moving to next step after {it} iterations, final error: {err}')
                    if err > 1.00:
                        raise ValueError(f'Change Iteration is not converging to reasonable value, try a smaller time step. Final iteration difference: {err}')
                    self.errorStag = 0
                    break
            self.err = err
        if it == self.params.maxIters - 1:
            print(f"⚠️ WARNING: Did not converge in {self.params.maxIters} iterations, final error: {err:.2e}")
        if self.params.iterationCheck == True:
            assert 0

    
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
        if not self.params.materialCoupled and self.grid.timeStep > 0:
            Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep-1], axis=0) / self.constants.c  
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
        elif self.params.materialCoupled and self.grid.timeStep > 0:
            Tmat = self.grid.temperatureSet[:, self.grid.timeStep]  
            Emat = self.problem.material.C_v(Tmat) * Tmat  
            Erad = np.sum(self.grid.fullTensorPhiTime[self.grid.timeStep-1], axis=0) / self.constants.c  
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
        if self.grid.timeStep == 0:
            print("Energy conservation check skipped for initial time step.")
            return True
        
        print(f"Energy conservation check passed at time step {self.grid.timeStep} (time={self.grid.timeSet[self.grid.timeStep]:.2e})")
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
        self.runTime = tock - tick
        self.dtMax = np.max(self.grid.dt)
        print(f'delta t max of: {np.max(self.grid.dt):.2e}')
        print(f"Solve completed in {tock - tick:.2f} seconds.")
        if self.params.saveResults:
            self.saveResults()
            print("Results saved successfully.")
        else:
            saveResults = input("Would you like to save the results? (y/n): ")
            if saveResults.lower() == 'y':
                self.params.runName = input("Enter a filename descripter for run: ")
                self.saveResults()
                print("Results saved successfully.")
            else:
                print("Results not saved.")
            
        return self.grid.fullTensorTime, self.grid  # Return the full time-dependent solution tensor
    
    def saveResults(self):
        filePrefix = self.params.fileFolder
        runName = self.params.runName
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"dataStash/{filePrefix}/{runName}_{timestamp}.h5"
        fullTensorPhiTime = np.squeeze(self.grid.fullTensorPhiTime)[:-1,:,:]
        with h5py.File(filepath, "w") as f:
            f.create_dataset("fullTensorPhi", data=fullTensorPhiTime, compression="gzip")
            f.create_dataset("temperatureSet", data=self.grid.temperatureSet, compression="gzip")
            f.create_dataset("timeSet", data=self.grid.timeSet, compression="gzip")
            f.attrs["spaceGrid"] = self.grid.spaceGrid
            f.attrs["spaceMid"] = self.grid.spaceMid
            f.attrs["freqGrid"] = self.grid.freqGrid
            f.attrs["dt"] = self.grid.dt
            f.attrs["dx"] = self.grid.dx
            f.attrs["nBins"] = self.grid.nBins
            f.attrs["nSteps"] = self.params.nSteps
            f.attrs["sn"] = self.params.sn
            f.attrs["maxFreq"] = self.params.maxFreq
            f.attrs['runTime'] = self.runTime
            f.attrs["runLabel"] = self.params.runLabel

    




    


    
