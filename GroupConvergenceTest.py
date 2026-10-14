"""Group-convergence test: standard multigroup (MG) vs. variable-basis multigroup (VMG).

Infinite, homogeneous medium (two cells, reflective on both sides) with a temperature-independent,
frequency-dependent opacity sigma(nu) = SIGMA0 / (1 + (nu/NU0)^3). The radiation starts as a diluted
Planckian (color T_COLOR, energy a*T_RAD0^4) out of equilibrium with the material at T_MAT0.

Both methods are run through the production solver (Base.Base + Logic) for increasing group counts and
compared to a fine-frequency reference integrated to tight tolerance with solve_ivp. As G grows, both
should approach the reference (down to the O(dt) backward-Euler floor) and therefore each other.
"""
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import Base
import InfiniteMedium
import InfiniteVariable
import Logic as Log

SIGMA0 = 10.0       # cm^-1
NU0 = 1.0           # keV
CV = 0.01
T_MAT0, T_RAD0, T_COLOR = 0.4, 0.5, 1.0
MIN_FREQ, MAX_FREQ, INF_FREQ = 1e-4, 25.0, 125.0
T_MAX, N_STEPS = 0.2, 400
GROUP_SET = [8, 16, 32, 64, 128]
N_REF_GROUPS = 1500
N_GAUSS = 8

const = Base.Constants()
_xg, _wg = np.polynomial.legendre.leggauss(N_GAUSS)


def sigma_nu(nu):
    return SIGMA0 / (1.0 + (nu / NU0) ** 3)


def planck_nu(nu, T):
    return 15.0 * const.a * const.c / (4.0 * np.pi**5) * nu**3 / np.expm1(nu / T)


def dplanck_dT(nu, T):
    x = nu / T
    return planck_nu(nu, T) * x / (T * -np.expm1(-x))


def group_quad(f, lo, hi):
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    return half * sum(w * f(mid + half * x) for x, w in zip(_xg, _wg))


class TestMaterial:
    def __init__(self, params, grid, variable):
        self.params = params
        self.grid = grid
        self.variable = variable

    def sigma_a(self, freq, T):
        T = np.broadcast_to(T, (self.params.nBins,))
        lo = self.grid.freqGrid[:-1, None]
        hi = self.grid.freqGrid[1:, None]
        if self.variable:
            # u-group edges map to nu = u*T in each cell
            lo, hi = lo * T, hi * T
        num = group_quad(lambda nu: sigma_nu(nu) * planck_nu(nu, T), lo, hi)
        den = group_quad(lambda nu: planck_nu(nu, T), lo, hi)
        return num / den

    def C_v(self, T):
        return CV


class Problem:
    def __init__(self, grid, params, material):
        self.parameters = params
        self.material = material
        self.equations = Log.Logic(params, grid, material, const)
        self.equations.applyInitialConditions()


def make_params(variable, G):
    module = InfiniteVariable if variable else InfiniteMedium
    p = module.Parameters(tol=1e-10, maxIters=200, nSteps=N_STEPS)
    p.sn = 2
    p.nBins = 2
    p.xMin, p.xMax = -1.0, 1.0
    p.boundaryLeft = p.boundaryRight = "Reflective"
    p.groupSpace = "log"
    p.freqNum = G
    p.minFreq, p.maxFreq, p.infFreq = MIN_FREQ, MAX_FREQ, INF_FREQ
    p.matchNu = variable
    p.timeMax = T_MAX
    p.timeScale = "linear"
    p.splitStepsBool = False
    p.initialTemperature = T_MAT0
    p.radiationTemperature = T_RAD0
    p.colorTemperature = T_COLOR
    p.checkEnergy = False
    p.saveResults = False
    return p


def run_method(variable, G, time_set=None):
    params = make_params(variable, G)
    if time_set is not None:
        params.nSteps = len(time_set) - 1
        params.timeMax = time_set[-1]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        grid = Base.Grid(params, const)
        if time_set is not None:
            grid.timeSet = np.asarray(time_set, dtype=float)
            grid.dt = np.diff(grid.timeSet)
        problem = Problem(grid, params, TestMaterial(params, grid, variable))
        solver = Base.Base(grid, problem, params, const)
        E0 = CV * T_MAT0 + np.sum(problem.equations.getPhi()[:, 0]) / const.c
        grid.fullTensorPhiTime[0] = problem.equations.getPhi()
        grid.fullTensOld = grid.fullTensor.copy()
        # Same as Base.solve() minus the interactive save prompt
        for index in range(params.nSteps):
            solver.index = index
            solver.converge()
            solver.updateAll(index)
    phi = problem.equations.getPhi()
    T = grid.temperatureSet
    edges = grid.freqGrid * (T[0, -1] if variable else 1.0)
    E1 = CV * T[0, -1] + np.sum(phi[:, 0]) / const.c
    return {
        "t": grid.timeSet,
        "T": T[0],
        "Trad": (np.sum(grid.fullTensorPhiTime[:, :, 0], axis=1) / (const.a * const.c)) ** 0.25,
        "nonuniform": np.max(np.abs(T[0] - T[1])),
        "edges": edges,
        "phi": phi[:, 0],
        "energyDrift": abs(E1 - E0) / E0,
        "notConverged": log.getvalue().count("Did not converge"),
    }


def reference_solution(t_max=T_MAX):
    edges = np.geomspace(MIN_FREQ, INF_FREQ, N_REF_GROUPS + 1)
    lo, hi = edges[:-1], edges[1:]
    # Weighting is irrelevant at this resolution, so use a flat group average
    sig = group_quad(sigma_nu, lo, hi) / (hi - lo)
    Bg = lambda T: group_quad(lambda nu: planck_nu(nu, T), lo, hi)
    dBg = lambda T: group_quad(lambda nu: dplanck_dT(nu, T), lo, hi)

    phi0 = 4.0 * np.pi * Bg(T_COLOR)
    phi0 *= const.a * T_RAD0**4 / (np.sum(phi0) / const.c)
    y0 = np.append(phi0, T_MAT0)
    G = N_REF_GROUPS
    idx = np.arange(G)

    def rhs(t, y):
        phi, T = y[:-1], y[-1]
        emis = 4.0 * np.pi * Bg(T)
        return np.append(const.c * sig * (emis - phi), np.sum(sig * (phi - emis)) / CV)

    def jac(t, y):
        dE = 4.0 * np.pi * dBg(y[-1])
        rows = np.concatenate([idx, idx, np.full(G, G), [G]])
        cols = np.concatenate([idx, np.full(G, G), idx, [G]])
        data = np.concatenate([-const.c * sig, const.c * sig * dE, sig / CV, [-np.sum(sig * dE) / CV]])
        return sparse.csc_matrix((data, (rows, cols)), shape=(G + 1, G + 1))

    sol = solve_ivp(rhs, (0.0, t_max), y0, method="BDF", jac=jac,
                    rtol=1e-9, atol=1e-14, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return edges, sol


def cumulative_error(run, ref_edges, ref_phi):
    """Max error in the cumulative spectrum int_0^nu phi dnu', normalized by the total."""
    ref_cum = np.concatenate([[0.0], np.cumsum(ref_phi)])
    run_cum = np.concatenate([[0.0], np.cumsum(run["phi"])])
    return np.max(np.abs(run_cum - np.interp(run["edges"], ref_edges, ref_cum))) / ref_cum[-1]


def main():
    print("Computing reference solution...")
    ref_edges, ref = reference_solution()
    ref_phi_final = ref.y[:-1, -1]

    results = {"MG": [], "VMG": []}
    header = f"{'method':>5} {'G':>5} {'max|dT|':>10} {'spec err':>10} {'E drift':>10} {'T(x) spread':>12} {'unconv':>7}"
    print(header)
    for G in GROUP_SET:
        for name, variable in (("MG", False), ("VMG", True)):
            run = run_method(variable, G)
            T_ref = ref.sol(run["t"])[-1]
            run["Terr"] = np.max(np.abs(run["T"] - T_ref))
            run["specErr"] = cumulative_error(run, ref_edges, ref_phi_final)
            results[name].append(run)
            print(f"{name:>5} {G:>5} {run['Terr']:10.3e} {run['specErr']:10.3e} "
                  f"{run['energyDrift']:10.3e} {run['nonuniform']:12.3e} {run['notConverged']:7d}")
    for G, mg, vmg in zip(GROUP_SET, results["MG"], results["VMG"]):
        print(f"G={G:4d}  max|T_MG - T_VMG| = {np.max(np.abs(mg['T'] - vmg['T'])):.3e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    t_fine = np.linspace(0.0, T_MAX, 500)
    ax1.plot(t_fine, ref.sol(t_fine)[-1], "k", lw=2, label="Reference")
    for G, mg, vmg in zip(GROUP_SET, results["MG"], results["VMG"]):
        line, = ax1.plot(mg["t"], mg["T"], lw=1, label=f"MG G={G}")
        ax1.plot(vmg["t"], vmg["T"], "--", color=line.get_color(), lw=1, label=f"VMG G={G}")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("Material temperature (keV)")
    ax1.legend(fontsize=7, ncol=2)

    for name, marker in (("MG", "o-"), ("VMG", "s--")):
        ax2.loglog(GROUP_SET, [r["Terr"] for r in results[name]], marker, label=f"{name}: max |T - T_ref|")
        ax2.loglog(GROUP_SET, [r["specErr"] for r in results[name]], marker, mfc="none",
                   label=f"{name}: final cumulative spectrum")
    ax2.set_xlabel("Number of groups")
    ax2.set_ylabel("Error")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/GroupConvergence.pdf")
    plt.show()
    return results


def equilibrium_plot(G=128, t_max=50.0, dt_max=0.1):
    """Material and radiation temperatures relaxing to equilibrium for MG and VMG vs. the reference."""
    # Unlinearized material iteration diverges for dt >~ 0.2 ns in this problem, so cap the step
    time_set = np.concatenate([[0.0], np.geomspace(1e-5, 1.0, 200), np.arange(1.0 + dt_max, t_max + 1e-9, dt_max)])
    _, ref = reference_solution(t_max)
    E0 = CV * T_MAT0 + np.sum(ref.y[:-1, 0]) / const.c
    T_eq = brentq(lambda T: CV * T + const.a * T**4 - E0, T_MAT0, T_COLOR)

    t_ref = np.geomspace(1e-5, t_max, 1000)
    y_ref = ref.sol(t_ref)
    curves = {"Reference": (t_ref, y_ref[-1], (np.sum(y_ref[:-1], axis=0) / (const.a * const.c)) ** 0.25)}
    for name, variable in (("MG", False), ("VMG", True)):
        print(f"Running {name} with {G} groups...")
        run = run_method(variable, G, time_set)
        curves[name] = (run["t"][1:], run["T"][1:], run["Trad"][1:])

    styles = {"Reference": dict(lw=4, alpha=0.3, ls="-"),
              "MG": dict(lw=1.2, ls="-"),
              "VMG": dict(lw=1.2, ls="--")}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for name, (t, Tm, Tr) in curves.items():
        for T, color, sym in ((Tm, "C0", "T_m"), (Tr, "C3", "T_r")):
            ax1.semilogx(t, T, color=color, label=f"{name} ${sym}$", **styles[name])
            ax2.loglog(t, np.abs(T - T_eq), color=color, label=f"{name} ${sym}$", **styles[name])
    ax1.axhline(T_eq, color="gray", ls=":", label=f"$T_{{eq}}$ = {T_eq:.4f} keV")
    ax1.set_ylabel("Temperature (keV)")
    ax1.set_title(f"Relaxation to equilibrium, {G} groups")
    ax1.legend(fontsize=8, ncol=2)
    ax2.set_xlabel("t (ns)")
    ax2.set_ylabel(r"$|T - T_{eq}|$ (keV)")
    ax2.set_ylim(1e-8, 1e-1)
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"figures/Equilibrium_{G}groups.pdf")
    plt.show()
    return curves, T_eq


if __name__ == "__main__":
    main()
    equilibrium_plot()
