import numpy as np
try:
    from numba import njit
except ImportError:  # Fall back to plain Python (slow, but runs without numba)
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function
from scipy.integrate import solve_ivp

import Base as Base

DEFAULT_TEMPERATURE_OFFSET = 0.4

_PLANCK_GAUSS_NODES = np.array(
    [
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346424956498,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ],
    dtype=np.float64,
)
_PLANCK_GAUSS_WEIGHTS = np.array(
    [
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ],
    dtype=np.float64,
)


@njit
def _planckDimensionlessValue(z):
    if z <= 0.0:
        return 0.0
    if z < 1e-5:
        return z**2 * (1.0 - z / 2.0 + z**2 / 12.0)
    if z >= 50.0:
        return 0.0
    expNegative = np.exp(-z)
    return z**3 * expNegative / (1.0 - expNegative)


@njit
def _planckDimensionlessIntegral(zLo, zHi):
    # Composite quadrature resolves the peak; the Wien tail past 50 is negligible.
    zLo = max(zLo, 0.0)
    zHi = min(zHi, 50.0)
    if zHi <= zLo:
        return 0.0

    segmentCount = max(1, int(np.ceil((zHi - zLo) / 2.0)))
    segmentWidth = (zHi - zLo) / segmentCount
    integral = 0.0
    for segment in range(segmentCount):
        segmentLo = zLo + segment * segmentWidth
        midpoint = segmentLo + 0.5 * segmentWidth
        halfWidth = 0.5 * segmentWidth
        for nodeIndex in range(8):
            z = midpoint + halfWidth * _PLANCK_GAUSS_NODES[nodeIndex]
            integral += (
                halfWidth
                * _PLANCK_GAUSS_WEIGHTS[nodeIndex]
                * _planckDimensionlessValue(z)
            )
    return integral


@njit
def planckBarNumbaVariable(freqGrid, T, frequencyScale, a, c, h):
    nGroups = freqGrid.shape[0] - 1
    nBins = T.shape[0]
    out = np.zeros((nGroups, nBins))
    coefficient = (15.0 * a * c) / (4.0 * np.pi**5)

    for group in range(nGroups):
        for cell in range(nBins):
            temperature = T[cell]
            scale = frequencyScale[cell]
            if temperature <= 0.0 or scale <= 0.0:
                continue

            if freqGrid.ndim == 1:
                nuLo = freqGrid[group]
                nuHi = freqGrid[group + 1]
            else:
                nuLo = freqGrid[group, cell]
                nuHi = freqGrid[group + 1, cell]
            zLo = h * nuLo * scale / temperature
            zHi = h * nuHi * scale / temperature
            integral = _planckDimensionlessIntegral(zLo, zHi)
            out[group, cell] = (
                coefficient * (temperature / h) ** 4 * integral
            )
    return out


@njit
def planckBarDerivativeNumbaVariable(freqGrid, T, frequencyScale, a, c, h):
    nGroups = freqGrid.shape[0] - 1
    nBins = T.shape[0]
    out = np.zeros((nGroups, nBins))
    coefficient = (15.0 * a * c) / (4.0 * np.pi**5)

    for group in range(nGroups):
        for cell in range(nBins):
            temperature = T[cell]
            scale = frequencyScale[cell]
            if temperature <= 0.0 or scale <= 0.0:
                continue

            if freqGrid.ndim == 1:
                nuLo = freqGrid[group]
                nuHi = freqGrid[group + 1]
            else:
                nuLo = freqGrid[group, cell]
                nuHi = freqGrid[group + 1, cell]
            zLo = h * nuLo * scale / temperature
            zHi = h * nuHi * scale / temperature
            integral = _planckDimensionlessIntegral(zLo, zHi)
            boundaryTerm = (
                zLo * _planckDimensionlessValue(zLo)
                - zHi * _planckDimensionlessValue(zHi)
            )
            out[group, cell] = (
                coefficient
                * (temperature / h) ** 3
                * (4.0 * integral + boundaryTerm)
            )
    return out


def temperatureFromFourthPowerProfile(
    T0, profile, temperatureOffset=DEFAULT_TEMPERATURE_OFFSET
):
    """Convert an MMS T^4 profile with a uniform temperature floor to T."""
    fourthPower = (
        temperatureOffset**4 + np.asarray(T0) ** 4 * np.asarray(profile)
    )
    return np.maximum(fourthPower, 0.0) ** 0.25


def simpson(integrand, lo, hi):
    h = (hi - lo) / 3
    out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
    return out

def getPhi(w, psi):
    # Integrate over angles to get scalar flux
    fullTensorPhi = 4*np.pi*np.sum(w[:, None] * psi, axis=1)  # shape: (freqNum, nBins)
    return fullTensorPhi

# Base Planck definiton
def planck(nu, T):  # Planck function (not group integrated or weighted)
    const = Base.Constants
    nu, T = np.broadcast_arrays(np.asarray(nu), np.asarray(T))
    scaledFrequency = np.full(nu.shape, np.inf)
    np.divide(const.h * nu, T, out=scaledFrequency, where=T > 0)
    expNegative = np.exp(-scaledFrequency)
    denom = -np.expm1(-scaledFrequency)
    factor = np.divide(
        expNegative,
        denom,
        out=np.zeros_like(expNegative),
        where=denom > 0,
    )
    f = (15.0 * const.a * const.c) / (4.0 * np.pi**5)
    return f * nu**3 * factor

# Group integrated Planck
def planckBar(T, freqGrid):
    temperature = np.atleast_1d(np.asarray(T, dtype=float))
    result = planckBarNumbaVariable(
        np.asarray(freqGrid, dtype=float),
        temperature,
        np.ones_like(temperature),
        Base.Constants.a,
        Base.Constants.c,
        Base.Constants.h,
    )
    if np.asarray(T).ndim == 0 and np.asarray(freqGrid).ndim == 1:
        return result[:, 0]
    return result

# Planckian for opacity calculation
def planckg(T, freqGrid):
    # Calculate the Planck function for each frequency group
    # FIX: Broadcast frequency as column (freqNum, 1) against T (nBins,) -> Result is (freqNum, nBins)
    nu_lo = freqGrid[:-1] / T
    nu_hi = freqGrid[1:] / T
    integrand = lambda nu: (15.0 * nu**3) / np.pi**4 /  np.expm1(nu)
    bg = simpson(integrand, nu_lo, nu_hi)
    return bg  # Shape is now (freqNum, nBins)

def sigma_a(freqGrid, T, inputDict): 
    nu_lo = freqGrid[:-1]
    nu_hi = freqGrid[1:]
    sigma_aZero = 10 * np.ones(inputDict["freqNum"])
    denom = np.sqrt(T) * planckg(T, freqGrid)
    num = sigma_aZero * (np.exp(-nu_lo/T)-np.exp(-nu_hi/T))
    out = np.clip(num / denom, a_min=1e-4, a_max=1e8)
    # out = np.ones(inputDict["freqNum"]) * 1  # For testing purposes, set all opacities to a constant value
    return out

def MMSBase(
    tSet,
    T0,
    freqGrid,
    temperatureOffset=DEFAULT_TEMPERATURE_OFFSET,
):
    tau = 2
    decay = np.exp(-tau * tSet / 2)
    Trad = temperatureFromFourthPowerProfile(
        T0, 1 + decay, temperatureOffset
    )
    Tmat = temperatureFromFourthPowerProfile(
        T0, 1 - decay, temperatureOffset
    )
    return tSet, Trad, Tmat

def MMSSource(T0, T, TM1, t, dt , w, sigmaG, Cv, freqGrid):

    const = Base.Constants
    tau = 1
    c = np.exp((-tau * t)/2)
    Bg = planckBar(T, freqGrid)
    BgM1 = planckBar(TM1, freqGrid)

    # Constituent Parts
    dtBg = (Bg - BgM1) / dt
    dtT = tau * T0 / 2 * c
    dtPsi = const.c / 4 / np.pi * (Bg - dtBg * c + Bg * tau / 2 * c)
    phig = getPhi(w, freqGrid)
    psig = const.c / 4 /np.pi * (Bg) * (1 - c)

    # Sources
    S = 1 / const.c * dtPsi  - 4 * np.pi * sigmaG * Bg + sigmaG * psig
    Q = Cv * dtT - np.sum(sigmaG * phig - 4 * np.pi * sigmaG * Bg)

    return S, Q

def MMSSource1D(
    T0,
    T,
    TM1,
    t,
    dt,
    x,
    mu,
    w,
    sigmaG,
    Cv,
    freqGrid,
    sigmaX,
    tau=2,
    materialT=None,
    materialTPrevious=None,
    temperatureOffset=DEFAULT_TEMPERATURE_OFFSET,
):
    """Return Gaussian 1-D manufactured transport and material sources."""
    const = Base.Constants
    x = np.asarray(x)
    T = np.broadcast_to(np.asarray(T), x.shape)
    TM1 = np.broadcast_to(np.asarray(TM1), x.shape)
    mu = np.asarray(mu)
    w = np.asarray(w)
    sigmaG = np.asarray(sigmaG)
    if sigmaG.ndim == 1:
        sigmaG = sigmaG[:, None]

    expTerm = np.exp(-tau * t / 2)
    spatial = np.exp(-(x**2) / (2 * sigmaX**2)) / (sigmaX * np.sqrt(2 * np.pi))
    spatialDerivative = -x / sigmaX**2 * spatial

    def planckBarByCell(temperature):
        temperature = np.asarray(temperature, dtype=float)
        bbar = planckBarNumbaVariable(
            np.asarray(freqGrid, dtype=float),
            temperature,
            np.ones_like(temperature),
            const.a,
            const.c,
            const.h,
        )
        return bbar

    Bg = planckBarByCell(T)
    BgM1 = planckBarByCell(TM1)
    BgOffset = planckBarByCell(
        np.full(x.shape, temperatureOffset)
    )
    materialTemperature = np.broadcast_to(
        np.asarray(T if materialT is None else materialT), x.shape
    )
    BgMaterial = planckBarByCell(materialTemperature)
    dtBg = (Bg - BgM1) / dt

    psi = (
        Bg * (1 + expTerm) * spatial[None, :]
        + BgOffset
    )
    dtPsi = spatial[None, :] * (
        dtBg * (1 + expTerm) - tau / 2 * Bg * expTerm
    )
    dxPsi = (
        (1 + expTerm)
        * Bg
        * spatialDerivative[None, :]
    )

    psiByAngle = psi[:, None, :]
    phi = 4 * np.pi * np.sum(w[None, :, None] * psiByAngle, axis=1)
    sigmaByAngle = sigmaG[:, None, :]
    if materialTPrevious is None:
        materialTemperatureDerivative = (
            T0**4 * tau / 8 * expTerm * spatial
            / materialTemperature**3
        )
    else:
        previousTemperature = np.broadcast_to(
            np.asarray(materialTPrevious), x.shape
        )
        materialTemperatureDerivative = (
            materialTemperature - previousTemperature
        ) / dt

    S = (
        dtPsi[:, None, :] / const.c
        + mu[None, :, None] * dxPsi[:, None, :]
        - 4 * np.pi * sigmaByAngle * BgMaterial[:, None, :]
        + sigmaByAngle * psiByAngle
    )
    Q = Cv * materialTemperatureDerivative - np.sum(
        sigmaG * phi - 4 * np.pi * sigmaG * BgMaterial,
        axis=0,
    )
    return S, Q
