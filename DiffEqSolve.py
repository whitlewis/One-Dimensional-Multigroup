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

# # Planckian for opacity calculation
# def planckg(T, freqGrid):
#     # Calculate the Planck function for each frequency group
#     # FIX: Broadcast frequency as column (freqNum, 1) against T (nBins,) -> Result is (freqNum, nBins)
#     nu_lo = freqGrid[:-1] / T
#     nu_hi = freqGrid[1:] / T
#     integrand = lambda nu: (15.0 * nu**3) / np.pi**4 /  np.expm1(nu)
#     bg = simpson(integrand, nu_lo, nu_hi)
#     return bg  # Shape is now (freqNum, nBins)

# def sigma_a(freqGrid, T, inputDict): 
#     nu_lo = freqGrid[:-1]
#     nu_hi = freqGrid[1:]
#     sigma_aZero = 10 * np.ones(inputDict["freqNum"])
#     denom = np.sqrt(T) * planckg(T, freqGrid)
#     num = sigma_aZero * (np.exp(-nu_lo/T)-np.exp(-nu_hi/T))
#     out = np.clip(num / denom, a_min=1e-8, a_max=1e10)
#     # out = np.ones(inputDict["freqNum"]) * 1  # For testing purposes, set all opacities to a constant value
#     # out = np.ones(inputDict["freqNum"]) * 1 / 2 / T
#     return out


def sigmaAP(nu, T):
    sa0 = 10
    num = 1 - np.exp(-nu / T)
    denom = nu**3 * T*(1/2)
    return sa0 * num / denom

def planck(nu, T):  # Planck function (not group integrated or weighted)
    const = Base.Constants
    denom = np.expm1(const.h * nu/T)  # exp(x)-1 safely
    f = (15.0 * const.a * const.c) / (4.0 * np.pi**5)
    return f * nu**3 / denom

def sigma_a(freqGrid, T, inputDict):
    nu_lo = freqGrid[:-1]
    nu_hi = freqGrid[1:]
    num = lambda nu: planck(nu, T) * sigmaAP(nu, T)
    denom = lambda nu: planck(nu, T)
    numG = simpson(num, nu_lo, nu_hi)
    denomG = simpson(denom, nu_lo, nu_hi)
    out = numG / denomG
    # out = np.ones(inputDict["freqNum"]) * 1 / 2 / T
    return out
# End of opacity implementation

def planckVCM(u, T):  # Planck function for variable basis (not group integrated or weighted)
    const = Base.Constants
    denom = np.expm1(const.h * u)  # exp(x)-1 safely
    f = (15.0 * const.a * const.c) / (4.0 * np.pi**5)
    return f * u**3 * T**4 / denom

def sigmaAPV(u, T):
    sa0 = 10
    num = 1 - np.exp(-u)
    denom = u**3 * T**(5/2)
    return sa0 * num / denom 


def sigma_aVCM(freqGrid, T):
    u_lo = freqGrid[:-1, None]
    u_hi = freqGrid[1:, None]
    num = lambda u: planckVCM(u, T) * sigmaAPV(u, T)
    denom = lambda u: planckVCM(u, T)
    numG = simpson(num, u_lo, u_hi)
    denomG = simpson(denom, u_lo, u_hi)
    out = numG / denomG
    # out = np.ones((self.params.freqNum, self.params.nBins)) * 1 / 2 / T  # For testing purposes, set all opacities to a constant value
    return out


def IMProblem(t, y, freqGrid, sigmaFunction, inputDict):
    const = Base.Constants

    T = y[-1]
    phi = y[:-1]
    C_v = .01

    planckSet = planckBar(T, freqGrid)
    sigmaSet = sigmaFunction(freqGrid, T, inputDict)

    yOut = np.zeros(len(phi) + 1)
    yOut[:-1] = const.c * sigmaSet * (4 * np.pi * planckSet - phi)
    yOut[-1] = (np.sum(sigmaSet * phi) - 4* np.pi *np.sum(sigmaSet * planckSet)) / C_v
    return yOut




def IMSolve(inputDict):
    const = Base.Constants
    T = inputDict["initialTemperature"]
    Trad = inputDict["radiationTemperature"]
    Tc = inputDict["colorTemperature"]
    freqGrid = inputDict["freqGrid"]
    t_span = (0, inputDict["timeMax"])
    t_eval = inputDict["timeSet"]
    y0 = np.zeros(inputDict["freqNum"] + 1)
    Erad = const.a * const.c * T**4
    planckInit = 4 * np.pi * planckBar(Trad, freqGrid).flatten()
    colorInit = 4 * np.pi * planckBar(Tc, freqGrid).flatten()
    colorInit *= np.sum(planckInit) / np.sum(colorInit)
    freqGroups = .5* (freqGrid[1:] + freqGrid[:-1])
    # plt.plot(freqGroups, colorInit, label = "color", linestyle="--")
    # plt.plot(freqGroups, planckInit, label = "PlanckBar")
    # plt.xlim(0,15)
    # plt.legend()
    # plt.show()

    y0[:-1] = colorInit
    y0[-1] = T
    solve_Clark = lambda t, y: IMProblem(t, y, freqGrid, sigma_a, inputDict)
    print("Starting Solve")
    solution = solve_ivp(solve_Clark, t_span, y0, method='BDF', t_eval=t_eval, rtol=1e-6,
    atol=1e-10)
    return solution


def plotEnergy(inputDict, solution):
    const = Base.Constants
    Erad = np.sum(solution.y[:-1], axis=0) / const.c
    Emat = 0.01 * solution.y[-1]
    Etot = Erad + Emat

    plt.figure()
    plt.plot(inputDict['timeSet'], Erad, label="Radiation")
    plt.plot(inputDict['timeSet'], Emat, label="Material")
    plt.plot(inputDict['timeSet'], Etot, label="Total")
    plt.legend()
    plt.show()

def plotRad():
    minFreq = 1e-4
    maxFreq = 25
    infFreq = 125
    freqNum = 100
    timeMax = 1.0
    timeNum = 100000



    inputDictTest = {
        'colorTemperature' : 1.0,
        'initialTemperature': 0.4,
        "radiationTemperature": 0.5,
        "freqGrid": np.append(np.linspace(minFreq, maxFreq, freqNum), infFreq),
        'minFreq' : minFreq,
        'maxFreq' : maxFreq,
        'infFreq' : infFreq,
        'freqNum' : freqNum,
        'timeMax': timeMax,
        'timeNum': timeNum,
        'timeSet': np.geomspace(1e-14, timeMax, timeNum)
    }

    const = Base.Constants
    solution = IMSolve(inputDictTest)

    t = solution.t
    T = solution.y[-1]
    Tr = (np.sum(solution.y[:-1],axis=0)/ const.a / const.c)**.25
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, T, label='T(t)', color='blue')
    plt.plot(t, Tr, label='Tr(t)', color='red')
    plt.title("100 group problem", fontsize=16)
    plt.xlabel("Time t (ns)", fontsize=14)
    plt.ylabel("T (keV)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()
    minFreq = 1e-4
    maxFreq = 25
    infFreq = 125
    freqNum = 100
    timeMax = 1.0
    timeNum = 100000



inputDict = {}
freqGrid = np.linspace(0, 25, 100)
freqPlot = .5 * (freqGrid[:-1] + freqGrid[1:])
T = .5
VCM = sigma_aVCM(freqGrid, T)
MG = sigma_a(freqGrid, T, inputDict)
plt.plot(freqPlot, MG, label="MG")
plt.plot(freqPlot, VCM, label="VMG", linestyle="--")
plt.legend(fontsize=12)
plt.grid()
plt.show()
