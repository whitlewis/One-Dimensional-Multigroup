import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt

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

    lo = freqGrid[:-1, None]
    hi = freqGrid[1:, None]

    freqGroups = 0.5 * (
        freqGrid[:-1] + freqGrid[1:]
    )
    integrand = lambda nu: planck(nu, T)
    bbar = simpson(integrand, lo, hi)
    return bbar


def IMProblem(y, freqGrid, material):
    const = Base.Constants

    T = y[-1]
    phi = y[:-1]

    planckSet = planckBar(T, freqGrid)
    sigmaSet = material.sigma_a(freqGrid, T)

    yOut = np.zeros(len(phi))
    yOut[:-1] = const.c * sigmaSet * (4 * np.pi * planckSet - phi)
    yOut[-1] = const.c * (np.sum(sigmaSet * phi) - np.sum(sigmaSet * planckBar)) / material.C_v(T)
    return yOut




def IMSolve(params, grid, material):
    const = Base.Constants
    T = params.initialTemperature
    Trad = params.radiationTemperature
    freqGrid = grid.freqGrid

