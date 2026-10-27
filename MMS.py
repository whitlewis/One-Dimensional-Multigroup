import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import Base as Base

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
    # Integrate the Planck function over each frequency group to get group-averaged source

    lo = freqGrid[:-1]
    hi = freqGrid[1:]

    freqGroups = 0.5 * (
        freqGrid[:-1] + freqGrid[1:]
    )
    integrand = lambda nu: planck(nu, T)
    bbar = simpson(integrand, lo, hi)
    return bbar

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

def MMSBase(tSet, T0, freqGrid):
    const = Base.Constants
    tau = 2
    Trad = T0*(1 + np.exp((-tau * tSet)/2))
    Tmat = T0*(1 - np.exp((-tau * tSet)/2))
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
        temperature = temperature[None, :]
        positiveTemperature = temperature > 0
        safeTemperature = np.where(positiveTemperature, temperature, 1.0)
        if freqGrid.ndim == 1:
            lo = freqGrid[:-1, None]
            hi = freqGrid[1:, None]
        else:
            lo = freqGrid[:-1]
            hi = freqGrid[1:]
        bbar = simpson(
            lambda nu: planck(nu, safeTemperature), lo, hi
        )
        return np.where(positiveTemperature, bbar, 0.0)

    Bg = planckBarByCell(T)
    BgM1 = planckBarByCell(TM1)
    materialTemperature = np.broadcast_to(
        np.asarray(T if materialT is None else materialT), x.shape
    )
    BgMaterial = planckBarByCell(materialTemperature)
    dtBg = (Bg - BgM1) / dt

    psi = (
        const.c / (4 * np.pi)
        * Bg
        * (1 + expTerm)
        * spatial[None, :]
    )
    dtPsi = const.c / (4 * np.pi) * spatial[None, :] * (
        dtBg * (1 + expTerm) - tau / 2 * Bg * expTerm
    )
    dxPsi = (
        const.c / (4 * np.pi)
        * (1 + expTerm)
        * Bg
        * spatialDerivative[None, :]
    )

    psiByAngle = psi[:, None, :]
    phi = 4 * np.pi * np.sum(w[None, :, None] * psiByAngle, axis=1)
    sigmaByAngle = sigmaG[:, None, :]
    if materialTPrevious is None:
        materialGrowth = -np.expm1(-tau * t / 2)
        materialTemperatureDerivative = (
            T0
            * tau
            / 8
            * expTerm
            * spatial ** (1 / 4)
            / materialGrowth ** (3 / 4)
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

def MMSPlot():
    tMin, tMax = 0, 1.0
    tNum = 1000
    T0 = 0.25
    tSet = np.linspace(tMin, tMax, tNum)
    tSet, Trad, Tmat = MMSBase(tSet, T0, 0)
    plt.plot(tSet, Tmat, label="MMS Material")
    plt.plot(tSet, Trad, label="MMS Radiation Temperature")
    plt.xlabel("t (ns)")
    plt.ylabel("Temperature (keV)")
    plt.minorticks_on()

    plt.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True
    )

    plt.legend(
        frameon=False,
        loc="best"
    )
    plt.show()

def MMSPlot1D(
    T0=0.25,
    sigmaX=0.5,
    t=0.2,
    tau=2,
    xMin=-2.0,
    xMax=2.0,
    xNum=500,
    savePath=None,
):
    """Plot the Gaussian 1-D manufactured temperatures and spatial profile."""
    x = np.linspace(xMin, xMax, xNum)
    expTerm = np.exp(-tau * t / 2)
    spatial = np.exp(-(x**2) / (2 * sigmaX**2)) / (
        sigmaX * np.sqrt(2 * np.pi)
    )
    Trad = T0 * ((1 + expTerm) * spatial) ** 0.25
    Tmat = T0 * ((1 - expTerm) * spatial) ** 0.25

    fig, (temperatureAxis, spatialAxis) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7), constrained_layout=True
    )
    temperatureAxis.plot(x, Trad, label="Radiation temperature")
    temperatureAxis.plot(x, Tmat, label="Material temperature")
    temperatureAxis.set_ylabel("Temperature (keV)")
    temperatureAxis.set_title(f"1-D Gaussian MMS solution at t = {t:g} ns")
    temperatureAxis.legend(frameon=False, loc="best")
    temperatureAxis.minorticks_on()
    temperatureAxis.tick_params(
        which="both", direction="in", top=True, right=True
    )

    spatialAxis.plot(x, spatial, color="black", label="Gaussian f(x)")
    spatialAxis.set_xlabel("x (cm)")
    spatialAxis.set_ylabel("f(x)")
    spatialAxis.legend(frameon=False, loc="best")
    spatialAxis.minorticks_on()
    spatialAxis.tick_params(
        which="both", direction="in", top=True, right=True
    )
    plt.show()


def MMSPlot1DTimeEvolution(
    T0=0.25,
    sigmaX=0.5,
    tau=2,
    tMin=0.0,
    tMax=1.0,
    tNum=300,
    xMin=-2.0,
    xMax=2.0,
    xNum=300,
    savePath=None,
):
    """Plot radiation and material temperature over space and time."""
    x = np.linspace(xMin, xMax, xNum)
    t = np.linspace(tMin, tMax, tNum)
    spatial = np.exp(-(x**2) / (2 * sigmaX**2)) / (
        sigmaX * np.sqrt(2 * np.pi)
    )
    expTerm = np.exp(-tau * t / 2)
    Trad = T0 * ((1 + expTerm[:, None]) * spatial[None, :]) ** 0.25
    Tmat = T0 * ((1 - expTerm[:, None]) * spatial[None, :]) ** 0.25

    fig, (radiationAxis, materialAxis) = plt.subplots(
        2, 1, sharex=True, sharey=True, figsize=(8, 7),
        constrained_layout=True,
    )
    radiationPlot = radiationAxis.pcolormesh(
        x, t, Trad, shading="auto", cmap="inferno"
    )
    materialPlot = materialAxis.pcolormesh(
        x, t, Tmat, shading="auto", cmap="inferno"
    )

    radiationAxis.set_title("Radiation temperature")
    materialAxis.set_title("Material temperature")
    materialAxis.set_xlabel("x (cm)")
    radiationAxis.set_ylabel("t (ns)")
    materialAxis.set_ylabel("t (ns)")
    fig.colorbar(radiationPlot, ax=radiationAxis, label="Temperature (keV)")
    fig.colorbar(materialPlot, ax=materialAxis, label="Temperature (keV)")
    fig.suptitle("1-D Gaussian MMS temperature evolution")

    for axis in (radiationAxis, materialAxis):
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
    plt.show()

if __name__ == "__main__":
    MMSPlot1D()
    MMSPlot1DTimeEvolution()