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
    denom = np.expm1(const.h * nu/T)  # exp(x)-1 safely
    f = (15.0 * const.a * const.c) / (4.0 * np.pi**5)
    return f * nu**3 / denom

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

def MMSPlot():
    tSet, tMin, tMax = 0, 1.0
    tNum = 1000
    T0 = 0.5
    tSet = np.linspace(tMin, tMax, tNum)
    Trad, Tmat = MMSBase(tSet, T0, 0)
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
