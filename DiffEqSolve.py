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

    lo = freqGrid[:-1, None]
    hi = freqGrid[1:, None]

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
    nu_lo = freqGrid[:-1, None] / T
    nu_hi = freqGrid[1:, None] / T
    integrand = lambda nu: (15.0 * nu**3) / np.pi**4 /  np.expm1(nu)
    bg = simpson(integrand, nu_lo, nu_hi)
    return bg  # Shape is now (freqNum, nBins)

def sigma_a(freqGrid, T, inputDict): 
    nu_lo = freqGrid[:-1, None]
    nu_hi = freqGrid[1:, None]
    sigma_aZero = 10 * np.ones((inputDict["freqNum"], inputDict["nBins"]))
    denom = np.sqrt(T) * planckg()
    num = sigma_aZero * (np.exp(-nu_lo/T)-np.exp(-nu_hi/T))
    out = np.clip(num / denom, a_min=1e-4, a_max=1e8)
    # out = np.ones(out.shape) * 100  # For testing purposes, set all opacities to a constant value
    return out


def IMProblem(y, freqGrid, sigmaFunction):
    const = Base.Constants

    T = y[-1]
    phi = y[:-1]
    C_v = 0.01

    planckSet = planckBar(T, freqGrid)
    sigmaSet = sigmaFunction(freqGrid, T)

    yOut = np.zeros(len(phi))
    yOut[:-1] = const.c * sigmaSet * (4 * np.pi * planckSet - phi)
    yOut[-1] = const.c * (np.sum(sigmaSet * phi) - np.sum(sigmaSet * planckBar)) / C_v
    return yOut




def IMSolve(sigmaFunction, inputDict):
    const = Base.Constants
    T = inputDict["initialTemperature"]
    Trad = inputDict["radiationTemperature"]
    freqGrid = inputDict["freqGrid"]
    t_span = (0, inputDict["timeMax"])
    t_eval = inputDict["timeSet"]
    y0 = planckBar(Trad, freqGrid)

    solve_Clark = lambda y: IMProblem(y, freqGrid, sigmaFunction)
    solution = solve_ivp(solve_Clark, t_span, y0, method='BDF', t_eval=t_eval)


minFreq = 1e-4
maxFreq = 30
infFreq = 125
freqNum = 100

inputDictTest = {
    'initialTemperature': 0.4,
    "radiationTemperature": 0.5,
    "freqGrid": np.append(np.linspace(minFreq, maxFreq, freqNum), infFreq),
    'minFreq' : 1e-4,
    'maxFreq' : 30,
    'infFreq' : 125,
    'freqNum' : 100
}
const = Base.constants
solution = IMSolve(sigma_a, inputDictTest)
t = solution.t
T = solution.y[-1]
Tr = (np.sum(solution.y[0:100],axis=0)/ const.a)**.25
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, T, label='T(t)', color='blue')
plt.plot(t, Tr, label='Tr(t)', color='red')
plt.title("100 group problem", fontsize=16)
plt.xlabel("Time t (sh)", fontsize=14)
plt.ylabel("T (keV)", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()
