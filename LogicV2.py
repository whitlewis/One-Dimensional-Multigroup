import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import MMS


class SharedFunctions:
    def __init__(self, params, grid, constants, method):

        self.params = params
        self.grid = grid
        self.const = constants
        self.method = method

    def getPhi(self):
        # Integrate over angles to get scalar flux
        fullTensorPhi = 4*np.pi*np.sum(self.grid.w[:, None] * self.grid.fullTensor, axis=1)  # shape: (freqNum, nBins)
        return fullTensorPhi

    # Simpson for integration over group
    def simpson(self, integrand, lo, hi):
        h = (hi - lo) / 3
        out = 3/8 *h* (integrand(lo) + 3*integrand(lo + h) + 3*integrand(lo +2*h) + integrand(hi))
        return out

    # Group integrated Planck
    def planckBar(self, T):
        # Integrate the Planck function over each frequency group to get group-averaged source
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        freqGroups = 0.5 * (self.grid.freqGrid[:-1] + self.grid.freqGrid[1:])
        integrand = lambda nu: self.planck(nu, T)
        bbar = self.simpson(integrand, lo, hi)
        return bbar

    