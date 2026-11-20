import numpy as np
np.seterr(divide='raise', invalid='raise', over='raise')
import matplotlib.pyplot as plt
import MMS

# Optional numba kernels for the transport sweeps. Set params.useNumba = False to force the
# pure numpy/Python sweeps (identical results, slower).
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def useNumbaFor(params):
    return NUMBA_AVAILABLE and bool(getattr(params, "useNumba", True))


BOUNDARY_TYPES = ["Infinite", "Reflective", "Vacuum", "None", "Delta", "Planckian", "Bimodal"]


def deltaPhysicalFrequency(params, grid):
    """Physical photon energy of the bimodal delta spike (same units as the MG frequency grid).

    params.deltaFrequency sets it; the default is the centre of group freqNum//4 of the MG grid
    (the group the "Delta" boundary uses). VMG grids divided by T0 (matchNu) are mapped back.
    """
    value = getattr(params, "deltaFrequency", None)
    if value is not None:
        return float(value)
    centre = grid.freqGroups[params.freqNum // 4]
    if getattr(params, "movingCoordinates", False) and getattr(params, "matchNu", False):
        centre = centre * params.initialTemperature
    params.deltaFrequency = float(centre)   # record the value used (saved to the .h5)
    return params.deltaFrequency


def bimodalSpectrum(planckSpectrum, freqGrid, deltaCoordinate, fraction):
    """(1 - fraction) of the energy as the given group-integrated Planckian, plus `fraction` of it
    in the single group whose edges contain deltaCoordinate (clamped to the grid ends)."""
    planckSpectrum = np.asarray(planckSpectrum, dtype=float)
    out = (1.0 - fraction) * planckSpectrum
    group = int(np.clip(np.searchsorted(freqGrid, deltaCoordinate, side="right") - 1, 0, planckSpectrum.shape[0] - 1))
    out[group] += fraction * np.sum(planckSpectrum)
    return out


if NUMBA_AVAILABLE:
    @njit
    def sweepBlockNumba(rhs, sig_t, speed, incoming, forward):
        """Upwind spatial sweep. rhs (G, nA, nBins), sig_t (G, nBins), speed (nA,), incoming (G, nA)."""
        G, nAngles, nBins = rhs.shape
        out = np.empty((G, nAngles, nBins))
        for f in range(G):
            for a in range(nAngles):
                s = speed[a]
                upstream = incoming[f, a]
                for step in range(nBins):
                    i = step if forward else nBins - 1 - step
                    upstream = (rhs[f, a, i] + s * upstream) / (s + sig_t[f, i])
                    out[f, a, i] = upstream
        return out

    @njit
    def movingMeshSweepNumba(rhs, sigT, c, mu, dx, freqGrid, du, incoming):
        """Moving-mesh sweep: upwind in space, and upwind in u by the sign of c within each cell.
        rhs (G, sn, nBins), sigT (G, nBins), c (sn, nBins), incoming spatial boundary (G, sn)."""
        G, sn, nBins = rhs.shape
        out = np.empty((G, sn, nBins))
        upstream = np.empty(G)
        cur = np.empty(G)
        for m in range(sn):
            forward = mu[m] >= 0
            spatial_speed = abs(mu[m]) / dx
            for f in range(G):
                upstream[f] = incoming[f, m]
            for step in range(nBins):
                i = step if forward else nBins - 1 - step
                cc = c[m, i]
                if cc > 0:
                    for f in range(G - 1, -1, -1):
                        denominator = sigT[f, i]
                        numerator = rhs[f, m, i]
                        numerator += spatial_speed * upstream[f]
                        denominator += spatial_speed
                        denominator += cc * freqGrid[f] / du[f]
                        if f + 1 < G:
                            numerator += cc * freqGrid[f + 1] / du[f + 1] * cur[f + 1]
                        cur[f] = numerator / denominator
                elif cc < 0:
                    frequency_speed = -cc
                    for f in range(G):
                        denominator = sigT[f, i]
                        numerator = rhs[f, m, i]
                        numerator += spatial_speed * upstream[f]
                        denominator += spatial_speed
                        denominator += frequency_speed * freqGrid[f + 1] / du[f]
                        if f > 0:
                            numerator += frequency_speed * freqGrid[f] / du[f - 1] * cur[f - 1]
                        cur[f] = numerator / denominator
                else:
                    for f in range(G):
                        denominator = sigT[f, i]
                        numerator = rhs[f, m, i]
                        numerator += spatial_speed * upstream[f]
                        denominator += spatial_speed
                        cur[f] = numerator / denominator
                for f in range(G):
                    out[f, m, i] = cur[f]
                    upstream[f] = cur[f]
        return out


def manufacturedSpatialProfile(params, x):
    sigmaX = getattr(params, "mmsSigmaX", 0.5)
    return np.exp(-(x**2) / (2 * sigmaX**2)) / (
        sigmaX * np.sqrt(2 * np.pi)
    )


def manufacturedTemperatureProfile(params, grid, time):
    T0 = getattr(params, "mmsT0", params.radiationTemperature)
    tau = getattr(params, "mmsTau", 2.0)
    temperatureOffset = getattr(
        params, "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET
    )
    profile = (
        (1 - np.exp(-tau * time / 2))
        * manufacturedSpatialProfile(params, grid.spaceMid)
    )
    return MMS.temperatureFromFourthPowerProfile(
        T0, profile, temperatureOffset
    )


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
    temperatureOffset = getattr(
        params, "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET
    )
    offsetPlanck = MMS.simpson(
        lambda nu: MMS.planck(nu, temperatureOffset), lo, hi
    )
    targetIntensity = (
        (1 + decay) * spatial[None, :] * groupPlanck
        + offsetPlanck
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
            lambda nu: (
                MMS.planck(nu, T0) * (1 + decay) * xProfile
                + MMS.planck(nu, temperatureOffset)
            ),
            boundaryEdges[:-1],
            boundaryEdges[1:],
        )
        return boundaryPlanck

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
    if frequencyScale is None:
        sigmaG = material.sigma_a(grid.freqGroups, temperature)
    else:
        sigmaG = material.sigma_a(
            grid.freqGroups,
            temperature,
            frequencyScale=frequencyScale,
        )
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
        temperatureOffset=getattr(
            params, "mmsTemperatureOffset", MMS.DEFAULT_TEMPERATURE_OFFSET
        ),
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

        # Caches that remove repeated work inside one iteration / time step
        self._materialStateCache = None   # (T, bbar, sigma_a) built by the latest rhsUpdate
        self._manufacturedCache = None    # MMS material source; depends only on the time step here
        self._boundaryCache = None        # Non-MMS boundary Planckians; fixed for the whole run

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
        temperature = np.atleast_1d(np.asarray(T, dtype=float))
        return MMS.planckBarNumbaVariable(
            self.grid.freqGrid,
            temperature,
            np.ones_like(temperature),
            self.const.a,
            self.const.c,
            self.const.h,
        )

    # Group-integrated dB/dT
    def planckBarDeriv(self, T):
        temperature = np.atleast_1d(np.asarray(T, dtype=float))
        return MMS.planckBarDerivativeNumbaVariable(
            self.grid.freqGrid,
            temperature,
            np.ones_like(temperature),
            self.const.a,
            self.const.c,
            self.const.h,
        )


    # Function for initial Condition as Planckian (helper for initialCondition)  
    def initSpectra(self):
        T0 = self.params.radiationTemperature
        Tc = self.params.colorTemperature
        planck = self.planckBar(Tc)
        self.grid.fullTensor[:] = planck[:, None]
        return self.grid.fullTensor.copy()
    
    def energyInitialCondition(self):
        T0 = self.params.radiationTemperature
        groupEnergies = self.planckBar(T0)[:, 0]
        
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
            temperatureOffset = getattr(
                self.params,
                "mmsTemperatureOffset",
                MMS.DEFAULT_TEMPERATURE_OFFSET,
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
            offsetBbar = self.planckBar(
                np.full(self.params.nBins, temperatureOffset)
            )
            intensity = (
                (1 + decay) * spatial[None, :] * bbar
                + offsetBbar
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
        # Delta: all of the Planckian's energy in the group containing nu_delta (a Bimodal with fraction 1)
        if self.params.boundaryLeft == "Delta":
            spectrum = bimodalSpectrum(self.planckBar(T0)[:, 0], self.grid.freqGrid, deltaPhysicalFrequency(self.params, self.grid), 1.0)
            planckLeft = np.broadcast_to(spectrum[:, None], (self.params.freqNum, self.sn)).copy()
        if self.params.boundaryRight == "Delta":
            spectrum = bimodalSpectrum(self.planckBar(T1)[:, 0], self.grid.freqGrid, deltaPhysicalFrequency(self.params, self.grid), 1.0)
            planckRight = np.broadcast_to(spectrum[:, None], (self.params.freqNum, self.sn)).copy()
        # Bimodal: half (params.deltaFraction) of the Planckian's energy moved into a delta spike
        fraction = getattr(self.params, "deltaFraction", 0.5)
        if self.params.boundaryLeft == "Bimodal":
            spectrum = bimodalSpectrum(self.planckBar(T0)[:, 0], self.grid.freqGrid, deltaPhysicalFrequency(self.params, self.grid), fraction)
            planckLeft = np.broadcast_to(spectrum[:, None], (self.params.freqNum, self.sn)).copy()
        if self.params.boundaryRight == "Bimodal":
            spectrum = bimodalSpectrum(self.planckBar(T1)[:, 0], self.grid.freqGrid, deltaPhysicalFrequency(self.params, self.grid), fraction)
            planckRight = np.broadcast_to(spectrum[:, None], (self.params.freqNum, self.sn)).copy()
        for side in (self.params.boundaryLeft, self.params.boundaryRight):
            if side not in BOUNDARY_TYPES:
                raise ValueError(f"Boundary must be one of: {BOUNDARY_TYPES}")
        return planckLeft, planckRight

    # Possible time varying Boundary condition
    def boundaryCondition(self, side, time):
        if getattr(self.params, "MMSCompare", False):
            x = self.grid.spaceGrid[0 if side == "left" else -1]
            spatial = manufacturedSpatialProfile(self.params, x)
            T0 = getattr(
                self.params, "mmsT0", self.params.radiationTemperature
            )
            temperatureOffset = getattr(
                self.params,
                "mmsTemperatureOffset",
                MMS.DEFAULT_TEMPERATURE_OFFSET,
            )
            decay = np.exp(-getattr(self.params, "mmsTau", 2.0) * time / 2)
            return np.broadcast_to(
                (1 + decay) * spatial * self.planckBar(T0)
                + self.planckBar(temperatureOffset),
                (self.params.freqNum, self.sn),
            ).copy()
        if self._boundaryCache is None:
            self._boundaryCache = self.boundaryPlanck()
        left, right = self._boundaryCache
        if side == "left":
            return left
        elif side == "right":
            return right

    # Time term from discretization
    def timeAbsorption(self):
        if not self.params.transient:
            return 0.0
        return 1.0 / (self.const.c * self.grid.dt[self.grid.timeStep])

    # Start the iteration (and the time step on its first call). Called once per iteration from rhsUpdate.
    def startTimeStep(self):
        # Latest iterate: used for the convergence check and the right reflective boundary.
        # Sweeps rebind grid.fullTensor to a new array, so aliasing is safe and avoids two copies.
        self.grid.fullTensOld = self.grid.fullTensor
        self.fullTens = self.grid.fullTensor
        if self.time_step == self.grid.timeStep:
            return

        self.time_step = self.grid.timeStep
        self.psi_old = self.grid.fullTensor.copy()
        self.timeTerm = self.timeAbsorption()
        self._manufacturedCache = None

    # Group Planckian and opacity at T, reused from the latest rhsUpdate when T is unchanged
    def materialState(self, T):
        cache = self._materialStateCache
        if cache is not None and np.array_equal(cache[0], T):
            return cache[1], cache[2]
        return self.planckBar(T), self.material.sigma_a(self.grid.freqGroups, T)

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
        bbar, sa = self.materialState(T_iterative)  # Group-averaged Planckian and opacity (cached from rhsUpdate)
        dbbar = self.planckBarDeriv(T_iterative)

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
        bbarGroup = self.planckBar(T)  # Get the group-averaged Planckian for the next temperature
        saGroup = self.material.sigma_a(self.grid.freqGroups, T)
        self._materialStateCache = (T.copy(), bbarGroup, saGroup)
        shape = (self.params.freqNum, self.sn, self.params.nBins)
        bbarNext = np.broadcast_to(bbarGroup[:, None, :], shape)  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(saGroup[:, None, :], shape)  # shape: (freqNum, sn, nBins)
        sourceTerms = None
        if getattr(self.params, "MMSCompare", False):
            # Without a moving frequency scale the MMS material source depends only on the time step
            if self._manufacturedCache is None:
                self._manufacturedCache = manufacturedSourceTerms(
                    self.params, self.grid, self.material, self.mu, self.grid.w
                )
            sourceTerms = self._manufacturedCache
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
        self.grid.rhs = rhs
        self.sigmaStarVar = self.sigmaStar(T, saGroup)


    # Define modified opacity (pass sigmaA when it has already been evaluated at T)
    def sigmaStar(self, T, sigmaA=None):
        if sigmaA is None:
            sigmaA = self.material.sigma_a(self.grid.freqGroups, T)
        return sigmaA + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity

    # Upwind sweep over one direction for a block of angles.
    # rhs: (G, nAngles, nBins), sig_t: (G, nBins), speed: (nAngles,), incoming: (G, nAngles)
    def sweepBlock(self, rhs, sig_t, speed, incoming, forward):
        if useNumbaFor(self.params):
            return sweepBlockNumba(
                np.ascontiguousarray(rhs, dtype=np.float64),
                np.ascontiguousarray(sig_t, dtype=np.float64),
                np.ascontiguousarray(speed, dtype=np.float64),
                np.ascontiguousarray(incoming, dtype=np.float64),
                bool(forward),
            )
        return self.sweepBlockNumpy(rhs, sig_t, speed, incoming, forward)

    # Same sweep vectorized over groups and angles with numpy (used when numba is off)
    @staticmethod
    def sweepBlockNumpy(rhs, sig_t, speed, incoming, forward):
        out = np.empty_like(rhs)
        nBins = rhs.shape[-1]
        cells = range(nBins) if forward else range(nBins - 1, -1, -1)
        upstream = incoming
        for i in cells:
            upstream = (rhs[:, :, i] + speed * upstream) / (speed + sig_t[:, i, None])
            out[:, :, i] = upstream
        return out

    def radiationSweep(self):
        # rhsUpdate starts the step/iteration; only start here if it was skipped
        if self.time_step != self.grid.timeStep:
            self.startTimeStep()

        # Set up Boundary conditions at the time step (allows for variable boundary conditions)
        mms = getattr(self.params, "MMSCompare", False)
        boundaryTimeIndex = self.grid.timeStep + int(mms)
        time = self.grid.timeSet[boundaryTimeIndex]
        phibl = self.boundaryCondition("left", time)
        phibr = self.boundaryCondition("right", time)

        rhs = self.grid.rhs
        sig_t = self.sigmaStarVar
        newFull = np.empty_like(self.fullTens)
        reflMatrix = np.asarray(self.reflMatrix)

        # Angles are swept in the original order: mu <= 0 first (backward), then mu > 0 (forward),
        # so the left reflective boundary can use this iteration's outgoing flux.
        forward = self.mu > 0
        backward = ~forward

        # Backward sweep
        if np.any(backward):
            if mms:
                bVal = phibr[:, backward]
            elif self.params.boundaryRight == "Reflective":
                # Reflective BC: use the reflected angle's outgoing flux (previous iterate) as the incoming flux
                bVal = self.fullTens[:, reflMatrix[backward], -1]
            elif self.params.boundaryRight == "Vacuum":
                bVal = np.zeros((self.freq, np.count_nonzero(backward)))
            else:
                bVal = phibr[:, backward]
            newFull[:, backward, :] = self.sweepBlock(
                rhs[:, backward, :], sig_t, np.abs(self.mu[backward]) / self.grid.dx, bVal, forward=False
            )

        # Forward sweep
        if np.any(forward):
            if mms:
                bVal = phibl[:, forward]
            elif self.params.boundaryLeft == "Reflective":
                # Reflective BC: use the reflected angle's outgoing flux from this iteration
                bVal = newFull[:, reflMatrix[forward], 0]
            elif self.params.boundaryLeft == "Vacuum":
                bVal = np.zeros((self.freq, np.count_nonzero(forward)))
            else:
                bVal = phibl[:, forward]
            newFull[:, forward, :] = self.sweepBlock(
                rhs[:, forward, :], sig_t, self.mu[forward] / self.grid.dx, bVal, forward=True
            )

        # numba does not honour np.seterr, so check the result explicitly
        if not np.all(np.isfinite(newFull)):
            raise FloatingPointError("Non-finite intensity produced by the radiation sweep")

        # Set fullTensor to the updated Tensor
        self.grid.fullTensor = newFull



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
        self.frequencyReferenceTemperature = np.full(
            self.params.nBins, self.params.initialTemperature
        )

        # Caches that remove repeated work inside one iteration / time step
        self._materialStateCache = None       # (T, bbar, sigma_a) built by the latest rhsUpdate
        self._previousFrequencyScale = None   # frequency scale of T^n, fixed for the time step

    def frequencyScale(self, temperature):
        """Return the physical-frequency scale for this VMG coordinate grid."""
        temperature = np.asarray(temperature)
        if getattr(self.params, "matchNu", False):
            reference = np.maximum(
                self.frequencyReferenceTemperature, 1e-12
            )
            # At the reference state, scaled VMG edges equal the MG physical edges.
            return self.params.initialTemperature * temperature / reference
        return temperature

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
        scaledFrequency = self.const.h * u * Tcoord / T
        safeScaledFrequency = np.minimum(scaledFrequency, 700.0)
        denom = np.expm1(safeScaledFrequency)
        f = (15.0 * self.const.a * self.const.c) / (4.0 * np.pi**5)
        numerator = f * u**3 * Tcoord**4
        return np.divide(
            numerator,
            denom,
            out=np.zeros_like(numerator, dtype=float),
            where=scaledFrequency < 700.0,
        )

    # Group integrated Planck at temperature T in the coordinate u = nu/Tcoord
    def planckBarInit(self, T, Tcoord):
        temperature, frequencyScale = np.broadcast_arrays(
            np.asarray(T, dtype=float),
            np.asarray(Tcoord, dtype=float),
        )
        temperature = np.atleast_1d(temperature).copy()
        frequencyScale = np.atleast_1d(frequencyScale).copy()
        return MMS.planckBarNumbaVariable(
            self.grid.freqGrid,
            temperature,
            frequencyScale,
            self.const.a,
            self.const.c,
            self.const.h,
        )
    
    def planckBar(self, T, frequencyScale=None):
        if frequencyScale is None:
            frequencyScale = self.frequencyScale(T)
        return MMS.planckBarNumbaVariable(
            self.grid.freqGrid,
            T,
            frequencyScale,
            self.const.a,
            self.const.c,
            self.const.h
        )

    # Function for initial Condition as Planckian (helper for initialCondition)  
    def initSpectra(self):
        T0 = self.params.radiationTemperature
        Tc = self.params.colorTemperature
        frequencyScale = self.frequencyScale(
            self.grid.temperatureSet[:, 0]
        )
        planck = self.planckBarInit(Tc, frequencyScale)  # shape: (freqNum, nBins)
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
            self.frequencyReferenceTemperature = temperature.copy()
            spatial = manufacturedSpatialProfile(
                self.params, self.grid.spaceMid
            )
            decay = np.exp(
                -getattr(self.params, "mmsTau", 2.0) * time / 2
            )
            bbar = self.planckBarInit(
                T0, self.frequencyScale(temperature)
            )
            offsetBbar = self.planckBarInit(
                getattr(
                    self.params,
                    "mmsTemperatureOffset",
                    MMS.DEFAULT_TEMPERATURE_OFFSET,
                ),
                self.frequencyScale(temperature),
            )
            intensity = (
                (1 + decay) * spatial[None, :] * bbar
                + offsetBbar
            )
            return np.broadcast_to(
                intensity[:, None, :],
                (self.params.freqNum, self.sn, self.params.nBins),
            ).copy()
        self.frequencyReferenceTemperature = np.full(
            self.params.nBins, self.params.initialTemperature
        )
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
    def boundaryPlanck(self, frequencyScale=None):
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
            frequencyScale = self.frequencyScale(temperature)
            temperatureOffset = getattr(
                self.params,
                "mmsTemperatureOffset",
                MMS.DEFAULT_TEMPERATURE_OFFSET,
            )
            left = np.broadcast_to(
                (1 + decay) * spatialLeft
                * self.planckBarInit(T0, frequencyScale[0])
                + self.planckBarInit(
                    temperatureOffset, frequencyScale[0]
                ),
                (self.params.freqNum, self.sn),
            ).copy()
            right = np.broadcast_to(
                (1 + decay) * spatialRight
                * self.planckBarInit(T0, frequencyScale[-1])
                + self.planckBarInit(
                    temperatureOffset, frequencyScale[-1]
                ),
                (self.params.freqNum, self.sn),
            ).copy()
            return left, right
        # Only Planckian sides read these values, so skip the Planck integrals otherwise
        shape = (self.params.freqNum, self.sn)
        if frequencyScale is None:
            frequencyScale = self.frequencyScale(self.grid.T_next)
        planckLeft = planckRight = None
        if self.params.boundaryLeft == "Planckian":
            planckLeft = np.broadcast_to(self.planckBarInit(self.params.setLeftBoundaryTemp, frequencyScale[0]), shape)
        if self.params.boundaryRight == "Planckian":
            planckRight = np.broadcast_to(self.planckBarInit(self.params.setRightBoundaryTemp, frequencyScale[-1]), shape)
        if self.params.boundaryLeft in ("Bimodal", "Delta"):
            planckLeft = np.broadcast_to(self.bimodalBoundary(self.params.setLeftBoundaryTemp, frequencyScale[0], self.params.boundaryLeft)[:, None], shape)
        if self.params.boundaryRight in ("Bimodal", "Delta"):
            planckRight = np.broadcast_to(self.bimodalBoundary(self.params.setRightBoundaryTemp, frequencyScale[-1], self.params.boundaryRight)[:, None], shape)
        return planckLeft, planckRight

    # Bimodal incoming spectrum in the boundary cell's coordinates: the spike sits at a fixed physical
    # energy nu_delta, i.e. in the group containing u = nu_delta / scale, so it moves through the
    # VMG groups as the boundary cell's scale changes (MG keeps it in a fixed group).
    # "Delta" is the same with all of the energy in the spike.
    def bimodalBoundary(self, boundaryTemperature, cellScale, kind="Bimodal"):
        planckSpectrum = self.planckBarInit(boundaryTemperature, cellScale)[:, 0]
        deltaCoordinate = deltaPhysicalFrequency(self.params, self.grid) / cellScale
        fraction = 1.0 if kind == "Delta" else getattr(self.params, "deltaFraction", 0.5)
        return bimodalSpectrum(planckSpectrum, self.grid.freqGrid, deltaCoordinate, fraction)

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

    # Start the iteration (and the time step on its first call). Called once per iteration from rhsUpdate;
    # sigma* is built in rhsUpdate from the opacity it already evaluates.
    def startTimeStep(self, frequencyScale=None):
        if self.time_step != self.grid.timeStep:
            self.time_step = self.grid.timeStep
            self.psi_old = self.grid.fullTensor.copy()
            self.timeTerm = self.timeAbsorption()
            self._previousFrequencyScale = None

        if frequencyScale is None:
            frequencyScale = self.frequencyScale(self.grid.T_next)
        # Sweeps rebind grid.fullTensor to a new array, so aliasing is safe and avoids a copy
        self.grid.fullTensOld = self.grid.fullTensor
        self.fullTens = self.grid.fullTensor
        self.movingMeshConst = self.movingMeshConstant(frequencyScale)  # Update moving mesh constant for the iteration
        self.bcLeft, self.bcRight = self.boundaryPlanck(frequencyScale)


    def polyPredict(self):
        # Quadratic (Lagrange) extrapolation of T to t^{n+1} from the last three stored levels.
        # Valid for non-uniform steps; reduces to 3T^n - 3T^{n-1} + T^{n-2} for uniform dt.
        k = self.grid.timeStep
        times = self.grid.timeSet[k - 2:k + 1]
        target = self.grid.timeSet[k + 1]
        T_guess = np.zeros(self.params.nBins)
        for j in range(3):
            weight = 1.0
            for m in range(3):
                if m != j:
                    weight *= (target - times[m]) / (times[j] - times[m])
            T_guess += weight * self.grid.temperatureSet[:, k - 2 + j]
        return T_guess

    # Initial iterate for a new time step: the extrapolated temperature (only a starting guess;
    # the iteration then converges to the backward-Euler solution regardless of it)
    def predictTemperature(self):
        if not getattr(self.params, "extrapolateTemp", False) or self.grid.timeStep < 2:
            return
        predicted = self.polyPredict()
        if np.all(np.isfinite(predicted)) and np.all(predicted > 0):
            self.grid.T_next = predicted
            self.T_next = predicted.copy()

    def materialEquation(self):
            # Outer constant calc
            dt = self.grid.dt[self.grid.timeStep]
            f = dt / self.material.C_v(self.grid.temperatureSet[:, self.grid.timeStep]) 
            
            # Newton step about the current iterate (the extrapolated predictor is only the step's first iterate)
            T_iterative = self.grid.T_next
            T = self.grid.temperatureSet[:, self.grid.timeStep]
            phi = self.getPhi()  # Compute scalar flux by integrating over angles
            bbar, sa = self.materialState(T_iterative)  # Group-averaged Planckian and opacity (cached from rhsUpdate)
            dbbar = 4.0 * bbar / T_iterative  # B-hat_g scales as T^4 at fixed u

            # Newton step on the material equation with emission linearized about T_iterative
            residual = T - T_iterative + f * np.sum(sa * phi - 4*np.pi*sa * bbar, axis=0)
            if getattr(self.params, "MMSCompare", False):
                residual += f * self.manufacturedQ
            T_next = T_iterative + self.params.temperatureLearningRate * residual / (1.0 + f * 4*np.pi * np.sum(sa * dbbar, axis=0))
            if getattr(self.params, "MMSCompare", False):
                T_next = np.maximum(T_next, 1e-12)
            self.T_next = T_next.copy()
            self.grid.T_next = T_next.copy()

    # Group Planckian and opacity at T, reused from the latest rhsUpdate when T is unchanged
    def materialState(self, T):
        cache = self._materialStateCache
        if cache is not None and np.array_equal(cache[0], T):
            return cache[1], cache[2]
        frequencyScale = self.frequencyScale(T)
        bbar = self.planckBar(T, frequencyScale)
        sa = self.material.sigma_a(
            self.grid.freqGroups,
            T,
            frequencyScale=frequencyScale,
        )
        return bbar, sa

    # This function updates Q*
    def rhsUpdate(self):
        if self.time_step != self.grid.timeStep:
            self.predictTemperature()  # first iterate of a new step
        T = self.grid.T_next
        frequencyScale = self.frequencyScale(T)
        self.startTimeStep(frequencyScale)  # psi_old and timeTerm must belong to this step before building Q*
        bbarGroup = self.planckBar(T, frequencyScale)  # Get the group-averaged Planckian for the next temperature
        saGroup = self.material.sigma_a(
            self.grid.freqGroups,
            T,
            frequencyScale=frequencyScale,
        )
        self._materialStateCache = (T.copy(), bbarGroup, saGroup)
        shape = (self.params.freqNum, self.sn, self.params.nBins)
        bbarNext = np.broadcast_to(bbarGroup[:, None, :], shape)  # shape: (freqNum, sn, nBins)
        sa = np.broadcast_to(saGroup[:, None, :], shape)  # shape: (freqNum, sn, nBins)
        sourceTerms = manufacturedSourceTerms(
            self.params,
            self.grid,
            self.material,
            self.mu,
            self.grid.w,
            frequencyScale=frequencyScale,
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
                frequencyScale=frequencyScale,
                frequencyVelocity=self.movingMeshConst,
            )
            rhs = (
                4 * np.pi * sa * bbarNext
                + self.timeTerm * self.psi_old
                + sourceS
            )
        self.sigmaStarVar = self.sigmaStar(T, saGroup)
        self.grid.rhs = rhs

    def movingMeshConstant(self, frequencyScale=None):   # This function calculates the C hat constant for the moving mesh equations
        T = self.grid.T_next
        if frequencyScale is None:
            frequencyScale = self.frequencyScale(T)
        dt = self.grid.dt[self.grid.timeStep]
        if self.grid.timeStep > 0:
            # T^n is fixed for the whole step, so its scale is computed once per step
            if self._previousFrequencyScale is None:
                self._previousFrequencyScale = self.frequencyScale(
                    self.grid.temperatureSet[:, self.grid.timeStep]
                )
            previousFrequencyScale = self._previousFrequencyScale
        else:
            previousFrequencyScale = frequencyScale
        dScaleDt = (frequencyScale - previousFrequencyScale) / dt
        dScaleDx = np.gradient(frequencyScale, self.grid.dx)
        # shape: (sn, nBins)
        return 1/frequencyScale[None, :] * (
            (1 / self.const.c * dScaleDt)[None, :] + self.mu[:, None] * dScaleDx[None, :]
        )


    # Define modified opacity (pass sigmaA when it has already been evaluated at T)
    def sigmaStar(self, T, sigmaA=None):
        if sigmaA is None:
            sigmaA = self.material.sigma_a(
                self.grid.freqGroups,
                T,
                frequencyScale=self.frequencyScale(T),
            )
        return sigmaA + 1/self.const.c*1/self.grid.dt[self.grid.timeStep]  # modified opacity
    
    # Here begins helper functions for the moving mesh radiation sweep

    # Incoming spatial boundary intensity for every group of angle m on one side, shape (freqNum,)
    def spatialBoundary(self, m, side):
        if side == "left":
            bc_type = self.params.boundaryLeft
            spatial_idx = 0
            planck = self.bcLeft
        elif side == "right":
            bc_type = self.params.boundaryRight
            spatial_idx = -1
            planck = self.bcRight
        else:
            raise ValueError("Side must be 'left' or 'right'")

        if getattr(self.params, "MMSCompare", False):
            return planck[:, m]

        # A missing boundary condition means no incoming flux.
        if bc_type == "Reflective":
            return self.fullTens[:, self.reflMatrix[m], spatial_idx]
        if bc_type in ("Vacuum", "None", None):
            return np.zeros(self.freq)
        if bc_type in ("Planckian", "Bimodal", "Delta"):
            return planck[:, m]
        raise ValueError(f"Unsupported {side} boundary condition: {bc_type!r}")

    # Sweep one angle through space; within each cell, sweep groups in the upwind direction of c.
    # Pure-Python floats are used for the scalar recurrence (much faster than indexing numpy scalars).
    # The arithmetic is in the same order as before, so results are unchanged.
    # No flux enters through the top/bottom u faces (the old setGroupBoundaryValues always returned 0).
    def sweepAngle(self, m, freqGrid, du, sigT):
        G = self.freq
        nBins = self.params.nBins
        mu = self.mu[m]
        forward = mu >= 0
        spatial_speed = abs(mu) / self.grid.dx
        upstream = self.spatialBoundary(m, "left" if forward else "right").tolist()
        rhsM = self.grid.rhs[:, m, :].T.tolist()      # nBins x G
        cM = self.movingMeshConst[m].tolist()         # nBins
        descending = range(G - 1, -1, -1)
        ascending = range(G)
        out = [None] * nBins
        cells = range(nBins) if forward else range(nBins - 1, -1, -1)

        for i in cells:
            rhs = rhsM[i]
            sig_t = sigT[i]
            c = cM[i]
            cur = [0.0] * G
            if c > 0:
                for f in descending:
                    denominator = sig_t[f]
                    numerator = rhs[f]
                    numerator += spatial_speed * upstream[f]
                    denominator += spatial_speed
                    # The tensor stores group-integrated intensity; divide by the donor width
                    # to form the spectral density used in the frequency-face flux.
                    denominator += c * freqGrid[f] / du[f]
                    if f + 1 < G:
                        numerator += c * freqGrid[f + 1] / du[f + 1] * cur[f + 1]
                    cur[f] = numerator / denominator
            elif c < 0:
                frequency_speed = -c
                for f in ascending:
                    denominator = sig_t[f]
                    numerator = rhs[f]
                    numerator += spatial_speed * upstream[f]
                    denominator += spatial_speed
                    denominator += frequency_speed * freqGrid[f + 1] / du[f]
                    if f > 0:
                        numerator += frequency_speed * freqGrid[f] / du[f - 1] * cur[f - 1]
                    cur[f] = numerator / denominator
            else:
                for f in ascending:
                    denominator = sig_t[f]
                    numerator = rhs[f]
                    numerator += spatial_speed * upstream[f]
                    denominator += spatial_speed
                    cur[f] = numerator / denominator
            out[i] = cur
            upstream = cur
        return np.array(out).T   # (G, nBins)

    def radiationSweep(self):
            # rhsUpdate starts the step/iteration; only start here if it was skipped
            if self.time_step != self.grid.timeStep:
                self.startTimeStep()

            if useNumbaFor(self.params):
                # Incoming spatial boundary for every angle on its upwind side, shape (freqNum, sn)
                incoming = np.empty((self.freq, self.sn))
                for m in range(self.sn):
                    incoming[:, m] = self.spatialBoundary(m, "left" if self.mu[m] >= 0 else "right")
                newFull = movingMeshSweepNumba(
                    np.ascontiguousarray(self.grid.rhs, dtype=np.float64),
                    np.ascontiguousarray(self.sigmaStarVar, dtype=np.float64),
                    np.ascontiguousarray(self.movingMeshConst, dtype=np.float64),
                    np.ascontiguousarray(self.mu, dtype=np.float64),
                    float(self.grid.dx),
                    np.ascontiguousarray(self.grid.freqGrid, dtype=np.float64),
                    np.ascontiguousarray(self.grid.du, dtype=np.float64),
                    incoming,
                )
            else:
                freqGrid = self.grid.freqGrid.tolist()
                du = self.grid.du.tolist()
                sigT = self.sigmaStarVar.T.tolist()   # nBins x G

                # Initialize the next tensor
                newFull = np.empty_like(self.fullTens)

                # Loop through Angles
                for m in range(self.sn):
                    newFull[:, m, :] = self.sweepAngle(m, freqGrid, du, sigT)

            # Python floats and numba do not honour np.seterr, so check the result explicitly
            if not np.all(np.isfinite(newFull)):
                raise FloatingPointError("Non-finite intensity produced by the moving-mesh sweep")

            # Set fullTensor to the updated Tensor
            self.grid.fullTensor = newFull


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
