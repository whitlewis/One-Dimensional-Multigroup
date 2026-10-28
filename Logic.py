import numpy as np
np.seterr(divide='raise', invalid='raise', over='raise')
from numba import njit
import matplotlib.pyplot as plt
import MMS


def manufacturedSpatialProfile(params, x):
    sigmaX = getattr(params, "mmsSigmaX", 0.5)
    return np.exp(-(x**2) / (2 * sigmaX**2)) / (
        sigmaX * np.sqrt(2 * np.pi)
    )


def manufacturedTemperatureProfile(params, grid, time):
    T0 = getattr(params, "mmsT0", params.radiationTemperature)
    tau = getattr(params, "mmsTau", 2.0)
    return T0 * (
        (1 - np.exp(-tau * time / 2))
        * manufacturedSpatialProfile(params, grid.spaceMid)
    ) ** 0.25


def discreteManufacturedRadiationSource(
    params, grid, mu, opacity, bbar, previousIntensity,
    frequencyScale=None, frequencyVelocity=None,
):
    """Build the source that makes the analytic intensity satisfy the discrete sweep."""
    timeIndex = grid.timeStep
    time = grid.timeSet[timeIndex + 1]
    T0 = getattr(params, "mmsT0", params.radiationTemperature)
    tau = getattr(params, "mmsTau", 2.0)
    decay = np.exp(-tau * time / 2)
    spatial = manufacturedSpatialProfile(params, grid.spaceMid)
    if frequencyScale is None:
        edges = grid.freqGrid
    else:
        edges = grid.freqGrid[:, None] * frequencyScale[None, :]
    if np.ndim(edges) == 1:
        lo, hi = edges[:-1, None], edges[1:, None]
    else:
        lo, hi = edges[:-1], edges[1:]
    groupPlanck = MMS.simpson(lambda nu: MMS.planck(nu, T0), lo, hi)
    targetIntensity = (
        grid.constants.c / (4 * np.pi)
        * (1 + decay)
        * spatial[None, :]
        * groupPlanck
    )
    targetIntensity = np.broadcast_to(
        targetIntensity[:, None, :],
        (params.freqNum, params.sn, params.nBins),
    )

    leftScale = (
        None if frequencyScale is None else frequencyScale[0]
    )
    rightScale = (
        None if frequencyScale is None else frequencyScale[-1]
    )
    spatialLeft = manufacturedSpatialProfile(params, grid.spaceGrid[0])
    spatialRight = manufacturedSpatialProfile(params, grid.spaceGrid[-1])

    def boundaryIntensity(xProfile, scale):
        boundaryEdges = (
            grid.freqGrid
            if scale is None
            else grid.freqGrid * scale
        )
        boundaryPlanck = MMS.simpson(
            lambda nu: MMS.planck(nu, T0),
            boundaryEdges[:-1],
            boundaryEdges[1:],
        )
        return (
            grid.constants.c / (4 * np.pi)
            * (1 + decay)
            * xProfile
            * boundaryPlanck
        )

    leftBoundary = boundaryIntensity(spatialLeft, leftScale)
    rightBoundary = boundaryIntensity(spatialRight, rightScale)
    opacity = np.asarray(opacity)
    if opacity.ndim == 2:
        opacity = opacity[:, None, :]
    timeTerm = (
        1 / (grid.constants.c * grid.dt[timeIndex])
        if params.transient else 0.0
    )
    result = (opacity + timeTerm) * targetIntensity
    result -= timeTerm * previousIntensity

    for angle, direction in enumerate(mu):
        spatialSpeed = abs(direction) / grid.dx
        if direction > 0 or (
            frequencyVelocity is not None and direction == 0
        ):
            upstream = np.empty_like(targetIntensity[:, angle, :])
            upstream[:, 0] = leftBoundary
            upstream[:, 1:] = targetIntensity[:, angle, :-1]
        else:
            upstream = np.empty_like(targetIntensity[:, angle, :])
            upstream[:, -1] = rightBoundary
            upstream[:, :-1] = targetIntensity[:, angle, 1:]
        result[:, angle, :] += spatialSpeed * (
            targetIntensity[:, angle, :] - upstream
        )

    if frequencyVelocity is not None:
        frequencyEdges = grid.freqGrid
        widths = np.diff(frequencyEdges)
        for group in range(len(widths)):
            velocity = frequencyVelocity
            positive = velocity > 0
            negative = velocity < 0
            groupSource = np.where(
                positive,
                velocity * frequencyEdges[group] / widths[group]
                * targetIntensity[group],
                0.0,
            )
            groupSource += np.where(
                negative,
                -velocity * frequencyEdges[group + 1] / widths[group]
                * targetIntensity[group],
                0.0,
            )
            if group + 1 < len(widths):
                groupSource -= np.where(
                    positive,
                    velocity * frequencyEdges[group + 1]
                    / widths[group + 1] * targetIntensity[group + 1],
                    0.0,
                )
            if group > 0:
                groupSource -= np.where(
                    negative,
                    -velocity * frequencyEdges[group] / widths[group - 1]
                    * targetIntensity[group - 1],
                    0.0,
                )
            result[group] += groupSource

    result -= 4 * np.pi * opacity * bbar
    return result


def manufacturedSourceTerms(params, grid, material, mu, w, frequencyScale=None):
    """Build MMS sources for the current time step when MMS comparison is enabled."""
    if not getattr(params, "MMSCompare", False):
        return None

    timeIndex = grid.timeStep
    time = grid.timeSet[timeIndex + 1]
    previousTime = grid.timeSet[timeIndex]
    T0 = getattr(params, "mmsT0", params.radiationTemperature)
    tau = getattr(params, "mmsTau", 2.0)
    x = grid.spaceMid
    spatial = manufacturedSpatialProfile(params, x)
    temperature = manufacturedTemperatureProfile(params, grid, time)
    previousTemperature = manufacturedTemperatureProfile(
        params, grid, previousTime
    )
    sigmaX = getattr(params, "mmsSigmaX", 0.5)
    sigmaG = material.sigma_a(grid.freqGroups, temperature)
    freqGrid = grid.freqGrid
    if frequencyScale is not None:
        freqGrid = freqGrid[:, None] * frequencyScale[None, :]

    baseTemperature = np.full(grid.nBins, T0)
    return MMS.MMSSource1D(
        T0,
        baseTemperature,
        baseTemperature,
        time,
        grid.dt[timeIndex],
        x,
        mu,
        w,
        sigmaG,
        material.C_v(temperature),
        freqGrid,
        sigmaX,
        tau=tau,
        materialT=temperature,
        materialTPrevious=previousTemperature,
    )


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
            self.grid.fullTensOld = self.grid.fullTensor.copy()
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
        self.manufacturedQ = np.zeros(self.params.nBins)
    
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
        nu, T = np.broadcast_arrays(np.asarray(nu), np.asarray(T))
        positiveTemperature = T > 0
        safeTemperature = np.where(positiveTemperature, T, 1.0)
        scaledFrequency = self.const.h * nu / safeTemperature
        expNegative = np.exp(-scaledFrequency)
        inverseDenominator = np.divide(
            expNegative,
            -np.expm1(-scaledFrequency),
            out=np.zeros_like(expNegative),
            where=scaledFrequency > 0,
        )
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return np.where(positiveTemperature, f * nu**3 * inverseDenominator, 0.0)

    # Group integrated Planck
    def planckBar(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source

        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]

        freqGroups = 0.5 * (
            self.grid.freqGrid[:-1] + self.grid.freqGrid[1:]
        )
        integrand = lambda nu: self.planck(nu, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar

    # Group-integrated dB/dT
    def planckBarDeriv(self, T):
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda nu: self.planck(nu, T) * (nu / T) / (T * -np.expm1(-nu / T))
        return self.simpson(integrand, lo, hi)


    # Function for initial Condition as Planckian (helper for initialCondition)  
    def initSpectra(self):
        T0 = self.params.radiationTemperature
        Tc = self.params.colorTemperature
        planck = self.planckBar(Tc)
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
        if getattr(self.params, "MMSCompare", False):
            time = self.grid.timeSet[0]
            T0 = getattr(
                self.params, "mmsT0", self.params.radiationTemperature
            )
            temperature = np.maximum(
                manufacturedTemperatureProfile(self.params, self.grid, time),
                1e-12,
            )
            self.T_next = temperature.copy()
            self.grid.T_next = temperature.copy()
            self.grid.temperatureSet[:, 0] = temperature
            spatial = manufacturedSpatialProfile(
                self.params, self.grid.spaceMid
            )
            decay = np.exp(
                -getattr(self.params, "mmsTau", 2.0) * time / 2
            )
            bbar = self.planckBar(np.full(self.params.nBins, T0))
            intensity = (
                self.const.c / (4 * np.pi)
                * (1 + decay)
                * spatial[None, :]
                * bbar
            )
            return np.broadcast_to(
                intensity[:, None, :],
                (self.params.freqNum, self.sn, self.params.nBins),
            ).copy()
        return self.initSpectra()
    
    # Helper function to define initial conditions
    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        if not getattr(self.params, "MMSCompare", False):
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
        if getattr(self.params, "MMSCompare", False):
            x = self.grid.spaceGrid[0 if side == "left" else -1]
            spatial = manufacturedSpatialProfile(self.params, x)
            T0 = getattr(
                self.params, "mmsT0", self.params.radiationTemperature
            )
            decay = np.exp(-getattr(self.params, "mmsTau", 2.0) * time / 2)
            return np.broadcast_to(
                self.const.c / (4 * np.pi)
                * (1 + decay)
                * spatial
                * self.planckBar(T0),
                (self.params.freqNum, self.sn),
            ).copy()
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
            # Latest iterate: used for the convergence check and the right reflective boundary
            self.grid.fullTensOld = self.grid.fullTensor.copy()
            self.fullTens = self.grid.fullTensor.copy()
            return

        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.fullTens = self.grid.fullTensor.copy()
        self.timeTerm = self.timeAbsorption()
        self.sigmaStarVar = self.sigmaStar(self.grid.T_next)  # Update sigma* for the time step

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
        dbbar = self.planckBarDeriv(T_iterative)
        sa = self.material.sigma_a(self.grid.freqGroups, T_iterative)
        
        # Newton step on the material equation with emission linearized about T_iterative
        residual = T - T_iterative + f * np.sum(sa * phi - 4*np.pi*sa * bbar, axis=0)
        if getattr(self.params, "MMSCompare", False):
            residual += f * self.manufacturedQ
        T_next = T_iterative + residual / (1.0 + f * 4*np.pi * np.sum(sa * dbbar, axis=0))
        if getattr(self.params, "MMSCompare", False):
            T_next = np.maximum(T_next, 1e-12)
        self.T_next = T_next.copy()
        self.grid.T_next = T_next.copy()

    def rhsUpdate(self):
        self.startTimeStep()  # psi_old and timeTerm must belong to this step before building Q*
        T = self.grid.T_next
        bbarNext = self.planckBar(T)  # Get the group-averaged Planckian for the next temperature
        bbarNext = np.broadcast_to(bbarNext[:, None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(self.material.sigma_a(self.grid.freqGroups, T)[:,None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sourceTerms = manufacturedSourceTerms(
            self.params, self.grid, self.material, self.mu, self.grid.w
        )
        if sourceTerms is None:
            rhs = sa * bbarNext + self.timeTerm * self.psi_old
            self.manufacturedQ.fill(0.0)
        else:
            _, self.manufacturedQ = sourceTerms
            sourceS = discreteManufacturedRadiationSource(
                self.params,
                self.grid,
                self.mu,
                sa,
                bbarNext,
                self.psi_old,
            )
            rhs = (
                4 * np.pi * sa * bbarNext
                + self.timeTerm * self.psi_old
                + sourceS
            )
        self.grid.rhs = rhs.copy()
        self.sigmaStarVar = self.sigmaStar(self.grid.T_next)


    # Define modified opacity
    def sigmaStar(self, T):
        return self.material.sigma_a(self.grid.freqGroups, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity

    def radiationSweep(self):
        # Initialize time set assets
        self.startTimeStep()

        # Set up Boundary conditions at the time step (allows for variable boundary conditions)
        boundaryTimeIndex = self.grid.timeStep + int(
            getattr(self.params, "MMSCompare", False)
        )
        time = self.grid.timeSet[boundaryTimeIndex]
        phibl = self.boundaryCondition("left", time)
        phibr = self.boundaryCondition("right", time)

        # if self.params.mms == True:
        #     S, Q = 
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
                    if getattr(self.params, "MMSCompare", False):
                        bVal = phibl[f, m]
                    elif self.params.boundaryLeft == "Reflective":
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

                    if getattr(self.params, "MMSCompare", False):
                        bVal = phibr[f, m]
                    elif self.params.boundaryRight == "Reflective":
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
        self.manufacturedQ = np.zeros(self.params.nBins)

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
    def planckVCM(self, u, T, Tcoord):  # Planck function for variable basis (not group integrated or weighted)
        # u = nu/Tcoord, so a Planckian at temperature T in this coordinate is Tcoord * B(u*Tcoord, T)
        denom = np.expm1(self.const.h * u * Tcoord / T)  # exp(x)-1 safely
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        return f * u**3 * Tcoord**4 / denom

    # Group integrated Planck at temperature T in the coordinate u = nu/Tcoord
    def planckBarInit(self, T, Tcoord):
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        integrand = lambda u: self.planckVCM(u, T, Tcoord)
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
        Tc = self.params.colorTemperature
        planck = self.planckBarInit(Tc, self.grid.temperatureSet[:, 0])  # shape: (freqNum, nBins)
        self.grid.fullTensor[:] = planck[:, None, :]
        return self.grid.fullTensor.copy()
    
    def energyInitialCondition(self):
        T0 = self.params.radiationTemperature

        # Integrate the VCM Planck function over the u-frequency groups
        groupEnergies = self.simpson(
            lambda u: self.planckVCM(u, T0, T0),
            self.grid.freqGrid[:-1],
            self.grid.freqGrid[1:]
        )

        # Expected radiation energy density per spatial cell
        EradExpectedPerCell = self.const.a * self.const.c * T0**4
        EradExpected = self.params.nBins * EradExpectedPerCell

        # Energy represented by the current angular intensity tensor
        currentEnergy = np.sum(self.getPhi())

        # Scale the tensor so that its total radiation energy is exactly a*T^4
        mult = EradExpected / currentEnergy
        self.grid.fullTensor *= mult

        # Check the resulting energy
        totalEnergy = np.sum(self.getPhi())

        # Analytical energy from the integrated VCM Planck function
        planckEnergyPerCell = (
            4.0 * np.pi
        ) * np.sum(groupEnergies)

        planckEnergy = planckEnergyPerCell * self.params.nBins

        # Radiation temperature after scaling
        radTemp = (
            totalEnergy /
            (self.params.nBins * self.const.a * self.const.c)
        )**0.25

        print(f"Analytical Planck energy: {planckEnergy:.6e}")
        print(f"Expected radiation energy: {EradExpected:.6e}")
        print(f"Actual initial energy: {totalEnergy:.6e}")
        print(f"Initial radiation temperature after scaling: {radTemp:.4e}")
        print(
            f"Planck init scaled by factor {mult:.4e} "
            f"to ensure radiation energy density corresponds to T = {T0:.4e}"
        )

        # Check analytical Planck integral
        if np.abs(planckEnergy - EradExpected) > self.params.energyTol:
            print(
                f"⚠️ Analytical VCM Planck energy "
                f"({planckEnergy:.4e}) does not match expected energy "
                f"({EradExpected:.4e})"
            )
    
    # Set initial Conditions here
    def initialCondition(self):
        if getattr(self.params, "MMSCompare", False):
            time = self.grid.timeSet[0]
            T0 = getattr(
                self.params, "mmsT0", self.params.radiationTemperature
            )
            temperature = np.maximum(
                manufacturedTemperatureProfile(self.params, self.grid, time),
                1e-12,
            )
            self.T_next = temperature.copy()
            self.grid.T_next = temperature.copy()
            self.grid.temperatureSet[:, 0] = temperature
            spatial = manufacturedSpatialProfile(
                self.params, self.grid.spaceMid
            )
            decay = np.exp(
                -getattr(self.params, "mmsTau", 2.0) * time / 2
            )
            bbar = self.planckBarInit(T0, temperature)
            intensity = (
                self.const.c / (4 * np.pi)
                * (1 + decay)
                * spatial[None, :]
                * bbar
            )
            return np.broadcast_to(
                intensity[:, None, :],
                (self.params.freqNum, self.sn, self.params.nBins),
            ).copy()
        return self.initSpectra()
    
    # Helper function to define initial conditions
    def applyInitialConditions(self):
        self.grid.fullTensor = self.initialCondition()
        if not getattr(self.params, "MMSCompare", False):
            self.energyInitialCondition()
        self.grid.fullTensorPhi = self.getPhi()
        self.fullTens = self.grid.fullTensor.copy()
        self.psi_old = self.grid.fullTensor.copy()


    # Incoming Planckians expressed in the boundary cells' current coordinates u = nu/T_cell
    def boundaryPlanck(self):
        if getattr(self.params, "MMSCompare", False):
            time = self.grid.timeSet[self.grid.timeStep + 1]
            T0 = getattr(
                self.params, "mmsT0", self.params.radiationTemperature
            )
            tau = getattr(self.params, "mmsTau", 2.0)
            decay = np.exp(-tau * time / 2)
            temperature = np.maximum(self.grid.T_next, 1e-12)
            spatialLeft = manufacturedSpatialProfile(
                self.params, self.grid.spaceGrid[0]
            )
            spatialRight = manufacturedSpatialProfile(
                self.params, self.grid.spaceGrid[-1]
            )
            factor = self.const.c / (4 * np.pi) * (1 + decay)
            left = np.broadcast_to(
                factor
                * spatialLeft
                * self.planckBarInit(T0, temperature[0]),
                (self.params.freqNum, self.sn),
            ).copy()
            right = np.broadcast_to(
                factor
                * spatialRight
                * self.planckBarInit(T0, temperature[-1]),
                (self.params.freqNum, self.sn),
            ).copy()
            return left, right
        shape = (self.params.freqNum, self.sn)
        planckLeft = np.broadcast_to(self.planckBarInit(self.params.setLeftBoundaryTemp, self.grid.T_next[0]), shape)
        planckRight = np.broadcast_to(self.planckBarInit(self.params.setRightBoundaryTemp, self.grid.T_next[-1]), shape)
        return planckLeft, planckRight

    # Possible time varying Boundary condition
    def boundaryCondition(self, side, time):
        if side == "left":
            return self.bcLeft
        elif side == "right":
            return self.bcRight

    # Time term from discretization
    def timeAbsorption(self):
        if not self.params.transient:
            return 0.0
        return 1.0 / (self.const.c * self.grid.dt[self.grid.timeStep])

    # Start the step
    def startTimeStep(self):
        if self.time_step == self.grid.timeStep:
            self.sigmaStarVar = self.sigmaStar(self.grid.T_next)  # Update sigma* for the iteration
            self.movingMeshConst = self.movingMeshConstant()  # Update moving mesh constant for the iteration
            self.bcLeft, self.bcRight = self.boundaryPlanck()
            self.grid.fullTensOld = self.grid.fullTensor
            self.fullTens = self.grid.fullTensor.copy()
            return

        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.timeTerm = self.timeAbsorption()
        self.sigmaStarVar = self.sigmaStar(self.grid.T_next)  # Update sigma* for the time step
        self.movingMeshConst = self.movingMeshConstant()  # Update moving mesh constant for the time step
        self.bcLeft, self.bcRight = self.boundaryPlanck()


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
            dbbar = 4.0 * bbar / T_iterative  # B-hat_g scales as T^4 at fixed u
            sa = self.material.sigma_a(self.grid.freqGroups, T_iterative)
            
            # Newton step on the material equation with emission linearized about T_iterative
            residual = T - T_iterative + f * np.sum(sa * phi - 4*np.pi*sa * bbar, axis=0)
            if getattr(self.params, "MMSCompare", False):
                residual += f * self.manufacturedQ
            T_next = T_iterative + self.params.temperatureLearningRate * residual / (1.0 + f * 4*np.pi * np.sum(sa * dbbar, axis=0))
            if getattr(self.params, "MMSCompare", False):
                T_next = np.maximum(T_next, 1e-12)
            self.T_next = T_next.copy()
            self.grid.T_next = T_next.copy()

    # This function updates Q*
    def rhsUpdate(self):
        self.startTimeStep()  # psi_old and timeTerm must belong to this step before building Q*
        T = self.grid.T_next
        bbarNext = self.planckBar(T)  # Get the group-averaged Planckian for the next temperature
        bbarNext = np.broadcast_to(bbarNext[:, None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(self.material.sigma_a(self.grid.freqGroups, T)[:,None,:], (self.params.freqNum, self.sn, self.params.nBins))  # shape: (freqNum, sn, nBins)
        sourceTerms = manufacturedSourceTerms(
            self.params,
            self.grid,
            self.material,
            self.mu,
            self.grid.w,
            frequencyScale=T,
        )
        if sourceTerms is None:
            rhs = sa * bbarNext + self.timeTerm * self.psi_old
            self.manufacturedQ.fill(0.0)
        else:
            _, self.manufacturedQ = sourceTerms
            sourceS = discreteManufacturedRadiationSource(
                self.params,
                self.grid,
                self.mu,
                sa,
                bbarNext,
                self.psi_old,
                frequencyScale=T,
                frequencyVelocity=self.movingMeshConst,
            )
            rhs = (
                4 * np.pi * sa * bbarNext
                + self.timeTerm * self.psi_old
                + sourceS
            )
        self.sigmaStarVar = self.sigmaStar(T)
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
        return self.material.sigma_a(self.grid.freqGroups, T) + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity
    
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

        if getattr(self.params, "MMSCompare", False):
            return self.boundaryCondition(side, time)[f, m]

        # A missing boundary condition means no incoming flux.
        if bc_type == "Reflective":
            bVal = self.fullTens[f, reflected_m, spatial_idx]
        elif bc_type in ("Vacuum", "None", None):
            bVal = 0.0
        elif bc_type == "Planckian":
            bVal = self.boundaryCondition(side, time)[f, m]
        elif bc_type == "Delta":
            if side == "left":
                T0 = self.params.setLeftBoundaryTemp
            else:
                T0 = self.params.setRightBoundaryTemp
            deltaSet = np.zeros((self.params.freqNum, self.sn))
            totalFlux = np.sum(self.grid.du * self.simpson(lambda nu: self.planckVCM(nu, T0, self.grid.T_next[spatial_idx]), self.grid.freqGrid[:-1], self.grid.freqGrid[1:]))
            deltaSet[self.params.freqNum//4, :] = totalFlux / self.grid.du[self.params.freqNum//4]      # Delta function at the middle frequency group
            bVal = deltaSet[f, m]
        else:
            raise ValueError(
                f"Unsupported {side} boundary condition: {bc_type!r}"
            )
                
        return bVal
    
    def setGroupBoundaryValues(self, f, m, c, newFull, time):
        bValGroup = 0.0
        return bValGroup
    
    def pickSweep(self, m, newWorkingSet, newFull):
        c = self.movingMeshConst[m, :]  # shape: (nBins,)

        if self.mu[m] >= 0:
            for i in range(self.params.nBins):
                spaceIndex = i - 1
                newWorkingSet = self.radiationBase(c, m, i, newWorkingSet, spaceIndex, newFull)
        elif self.mu[m] < 0:
            for i in range(self.params.nBins-1, -1, -1):
                spaceIndex = i + 1
                newWorkingSet = self.radiationBase(c, m, i, newWorkingSet, spaceIndex, newFull)

        return newWorkingSet
    
    def radiationBase(self, c, m, i, newWorkingSet, spaceIndex, newFull):
        rhs = self.grid.rhs[:, m, i]
        sig_t = self.sigmaStarVar[:, i]
        positive_c = c[i] > 0
        frequency_order = range(self.freq - 1, -1, -1) if positive_c else range(self.freq)
        spatial_speed = abs(self.mu[m]) / self.grid.dx
        is_spatial_boundary = (
            (self.mu[m] >= 0 and i == 0)
            or (self.mu[m] < 0 and i == self.params.nBins - 1)
        )
        if is_spatial_boundary:
            spatial_side = "left" if self.mu[m] >= 0 else "right"

        for f in frequency_order:
            denominator = sig_t[f]
            numerator = rhs[f]

            if is_spatial_boundary:
                spatial_boundary = self.setBoundaryValues(
                    f, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep], spatial_side
                )
                numerator += spatial_speed * spatial_boundary
            else:
                numerator += spatial_speed * newWorkingSet[f, spaceIndex]
            denominator += spatial_speed

            # The tensor stores group-integrated intensity; divide by the donor width
            # to form the spectral density used in the frequency-face flux.
            if positive_c:
                denominator += c[i] * self.grid.freqGrid[f] / self.grid.du[f]
                if f + 1 < self.freq:
                    donor_group = f + 1
                    numerator += (
                        c[i] * self.grid.freqGrid[f + 1]
                        / self.grid.du[donor_group]
                        * newWorkingSet[donor_group, i]
                    )
                else:
                    boundary_value = self.setGroupBoundaryValues(
                        f + 1, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep]
                    )
                    numerator += c[i] * self.grid.freqGrid[f + 1] * boundary_value
            elif c[i] < 0:
                frequency_speed = -c[i]
                denominator += (
                    frequency_speed * self.grid.freqGrid[f + 1] / self.grid.du[f]
                )
                if f > 0:
                    donor_group = f - 1
                    numerator += (
                        frequency_speed * self.grid.freqGrid[f]
                        / self.grid.du[donor_group]
                        * newWorkingSet[donor_group, i]
                    )
                else:
                    boundary_value = self.setGroupBoundaryValues(
                        f - 1, m, c[i], newFull, self.grid.timeSet[self.grid.timeStep]
                    )
                    numerator += frequency_speed * self.grid.freqGrid[f] * boundary_value

            newWorkingSet[f, i] = numerator / denominator
        return newWorkingSet
    

    def radiationSweep(self):
            # Initialize time set assets
            self.startTimeStep()

            # Set up Boundary conditions at the time step (allows for variable boundary conditions)
            boundaryTimeIndex = self.grid.timeStep + int(
                getattr(self.params, "MMSCompare", False)
            )
            time = self.grid.timeSet[boundaryTimeIndex]

            # Initialize the next tensor
            newFull = np.zeros_like(self.fullTens)

            # Loop through Angles
            for m in range(self.sn):
                newWorkingSet = newFull[:, m, :].copy()  # shape: (freqNum, nBins)
                newWorkingSet = self.pickSweep(m, newWorkingSet, newFull)
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
