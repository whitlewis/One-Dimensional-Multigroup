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
    tau = 0
    Trad = T0*(1 + np.exp((-tau * tSet)/2))
    Tmat = T0*(1 - np.exp((-tau * tSet)/2))
    psi = const.c / 4 /np.pi * (planckBar(Tmat, freqGrid)) * Tmat
