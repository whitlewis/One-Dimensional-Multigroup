import Marshak
from Marshak import Marshak as MarProblem
import InfiniteMedium   
from InfiniteMedium import InfiniteMedium as IM
import InfiniteVariable
from InfiniteVariable import InfiniteVariable as IV
import Reeds
from Reeds import Reeds as ReedsProblem
import Base as Base
import numpy as np
import Visualize as Vis

c = 29.9792458 # speed of light in cm/ns
a = 0.0137202
h = 1.0  # Frequency grid in keV

TMax = 1.5
TMin = .2
TNum = 50
TSet = np.linspace(TMin, TMax, TNum)

minFreq = 1e-4
maxFreq = 25
infFreq = 125
freqNum = 100
freqGrid = np.append(np.logspace(np.log10(minFreq), np.log10(maxFreq), freqNum), infFreq)

def simpson(integrand, lo, hi):
    h = (hi - lo) / 3
    out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) +integrand(hi))
    return out

# Base Planck definiton
def planckV(u, T):  # Planck function for variable basis (not group integrated or weighted)
    denom = np.expm1(h * u)  # exp(x)-1 safely
    f = (15.0 * a * c) / (4.0 * np.pi**5)
    return f * u**3 * T**4 / denom

# Group integrated Planck
def planckBarV(T, freqGrid):
    # Integrate the Planck function over each frequency group to get group-averaged source
    lo = freqGrid[:-1, None]
    hi = freqGrid[1:, None]
    integrand = lambda u: planckV(u, T)
    bbar = simpson(integrand, lo, hi)
    return bbar

# Base Planck definiton
def planck(nu, T):  # Planck function (not group integrated or weighted)
    denom = np.expm1(h * nu/T)  # exp(x)-1 safely
    f = (15.0 * a * c) / (4.0 * np.pi**5)
    return f * nu**3 / denom

# Group integrated Planck
def planckBar(T, freqGrid):
    # Integrate the Planck function over each frequency group to get group-averaged source
    lo = freqGrid[:-1, None]
    hi = freqGrid[1:, None]
    integrand = lambda nu: planck(nu, T)
    bbar = simpson(integrand, lo, hi)
    return bbar

def compareRank(TSet, freqGrid):
    variableMatrix = []
    for T in TSet:
        bbarV = np.append(planckBarV(T, freqGrid), T)
        variableMatrix.append(bbarV)

    multiGroupMatrix = []
    for T in TSet:
        bbar = np.append(planckBar(T, freqGrid), T)
        multiGroupMatrix.append(bbar)

    return np.array(variableMatrix), np.array(multiGroupMatrix)


def getRank(array):
    rank = np.linalg.matrix_rank(array)
    return rank


varaiableMatrix, multigroupMatrix = compareRank(TSet, freqGrid)

RVM = getRank(varaiableMatrix)
RGM = getRank(multigroupMatrix)

print(f'Rank of Moving coordinate Planckian: {RVM}')
print(f'Rank of Standard Planckian: {RGM}')


