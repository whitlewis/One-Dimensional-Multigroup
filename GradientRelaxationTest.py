"""Test C: closed slab relaxing with a smooth material-temperature gradient, MG vs. VMG.

Reflective walls on x in [-1, 1], material T0(x) = 0.4 + 0.1 sin(pi x / 2), spatially uniform dilute Planckian
radiation, and the opacity/C_v of GroupConvergenceTest. All runs share the spatial mesh, angular quadrature, and
time steps; the reference is MG with many groups. Unlike the infinite-medium test, VMG's mu*dT/dx stretching term
and the mismatch of u-grids between neighboring cells are active here.
"""
import contextlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np

import Base
import GroupConvergenceTest as gct

N_BINS = 40
SN = 4
T_MAX = 0.5
N_STEPS = 200
GROUP_SET = [8, 16, 32, 64, 128]
N_REF_GROUPS = 512
REF_FILE = f"dataStash/GradientRelaxation_MG{N_REF_GROUPS}_{N_BINS}cells.npz"


def initial_temperature(x):
    return 0.4 + 0.1 * np.sin(0.5 * np.pi * x)


def time_set():
    return np.concatenate([[0.0], np.geomspace(1e-4, T_MAX, N_STEPS)])


def run(variable, G, n_bins=N_BINS):
    params = gct.make_params(variable, G)
    params.nBins = n_bins
    params.sn = SN
    ts = time_set()
    params.nSteps = len(ts) - 1
    params.timeMax = T_MAX
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        grid = Base.Grid(params, gct.const)
        grid.timeSet = ts
        grid.dt = np.diff(ts)
        grid.temperatureSet[:, 0] = initial_temperature(grid.spaceMid)
        grid.T_next = grid.temperatureSet[:, 0].copy()
        problem = gct.Problem(grid, params, gct.TestMaterial(params, grid, variable))
        solver = Base.Base(grid, problem, params, gct.const)
        grid.fullTensorPhiTime[0] = problem.equations.getPhi()
        grid.fullTensOld = grid.fullTensor.copy()
        for index in range(params.nSteps):
            solver.index = index
            solver.converge()
            solver.updateAll(index)
    T = grid.temperatureSet.T.copy()                                   # (nt, nBins)
    Erad = grid.fullTensorPhiTime.sum(axis=1) / gct.const.c            # (nt, nBins)
    E = np.sum(gct.CV * T + Erad, axis=1) * grid.dx
    edges = grid.freqGrid[:, None] * (T[-1][None, :] if variable else np.ones((1, n_bins)))
    return {
        "t": ts,
        "x": grid.spaceMid,
        "T": T,
        "Trad": (Erad / gct.const.a) ** 0.25,
        "phi": grid.fullTensorPhiTime[-1],
        "edges": edges,
        "energyDrift": np.max(np.abs(E - E[0])) / E[0],
        "notConverged": log.getvalue().count("Did not converge"),
    }


def reference():
    if os.path.exists(REF_FILE):
        return dict(np.load(REF_FILE))
    print(f"Computing MG reference with {N_REF_GROUPS} groups (cached to {REF_FILE})...")
    ref = run(False, N_REF_GROUPS)
    os.makedirs(os.path.dirname(REF_FILE), exist_ok=True)
    np.savez(REF_FILE, **ref)
    return ref


def spectrum_error(run_out, ref):
    """Max over cells of the error in the final cumulative spectrum, normalized by the cell total."""
    err = 0.0
    for i in range(run_out["phi"].shape[1]):
        ref_cum = np.concatenate([[0.0], np.cumsum(ref["phi"][:, i])])
        cum = np.concatenate([[0.0], np.cumsum(run_out["phi"][:, i])])
        ref_at = np.interp(run_out["edges"][:, i], ref["edges"][:, i], ref_cum)
        err = max(err, np.max(np.abs(cum - ref_at)) / ref_cum[-1])
    return err


def main(plot_groups=16):
    ref = reference()
    results = {"MG": {}, "VMG": {}}
    print(f"{'method':>5} {'G':>5} {'max|dTm|':>10} {'max|dTr|':>10} {'spec err':>10} {'E drift':>10} {'unconv':>7}")
    for G in GROUP_SET:
        for name, variable in (("MG", False), ("VMG", True)):
            r = run(variable, G)
            r["Terr"] = np.max(np.abs(r["T"] - ref["T"]))
            r["Trerr"] = np.max(np.abs(r["Trad"] - ref["Trad"]))
            r["specErr"] = spectrum_error(r, ref)
            results[name][G] = r
            print(f"{name:>5} {G:>5} {r['Terr']:10.3e} {r['Trerr']:10.3e} {r['specErr']:10.3e} "
                  f"{r['energyDrift']:10.3e} {r['notConverged']:7d}")
    for G in GROUP_SET:
        print(f"G={G:4d}  max|T_MG - T_VMG| = {np.max(np.abs(results['MG'][G]['T'] - results['VMG'][G]['T'])):.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t = ref["t"]
    snap = [np.argmin(np.abs(t - ts)) for ts in (0.0, 0.01, 0.05, 0.5)]
    for ax, key, label in ((axes[0], "T", "Material"), (axes[1], "Trad", "Radiation")):
        for k, n in enumerate(snap):
            color = f"C{k}"
            ax.plot(ref["x"], ref[key][n], color=color, lw=4, alpha=0.3,
                    label=f"t = {t[n]:.2g} ns, MG {N_REF_GROUPS}")
            ax.plot(ref["x"], results["MG"][plot_groups][key][n], color=color, lw=1, label=f"MG {plot_groups}")
            ax.plot(ref["x"], results["VMG"][plot_groups][key][n], color=color, lw=1, ls="--",
                    label=f"VMG {plot_groups}")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel(f"{label} temperature (keV)")
    axes[0].legend(fontsize=7, ncol=2)

    for name, marker in (("MG", "o-"), ("VMG", "s--")):
        axes[2].loglog(GROUP_SET, [results[name][G]["Terr"] for G in GROUP_SET], marker,
                       label=f"{name}: max $|T_m - T_{{ref}}|$")
        axes[2].loglog(GROUP_SET, [results[name][G]["Trerr"] for G in GROUP_SET], marker, mfc="none",
                       label=f"{name}: max $|T_r - T_{{ref}}|$")
    axes[2].set_xlabel("Number of groups")
    axes[2].set_ylabel("Error vs. reference (keV)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/GradientRelaxation.pdf")
    plt.show()
    return results, ref


if __name__ == "__main__":
    main()
