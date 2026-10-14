# Solver Bug Report: One-Dimensional Multigroup / Variable-Basis Code

This report covers the code on branch `solver-bugfixes`. Bugs 1–4 are fixed on this branch. Items 5–11 are documented but **not** fixed.

Throughout, **MG** is the standard multigroup solver (`CoupledEquations` in `Logic.py`) and **VMG** is the variable-basis (moving-coordinate) solver (`MovingMeshEquations` in `Logic.py`).

## What is correct

Several errors in the write-up (*Multigroup First Crack*) do **not** appear in the code:

- The code uses $B$ as the Planck *intensity* consistently: the transport source is $\sigma B$, and the material equation uses $4\pi\sigma B$ and $\phi = 4\pi\sum_m w_m\psi_m$.
- The MG source is $Q^* = \sigma_g \bar B_g + \psi^n/(c\Delta t)$, and the $\mu<0$ sweep uses $|\mu|$.
- The VMG sweep implements the conservative form
  $$\frac{1}{c}\frac{\partial\hat\psi}{\partial t} + \mu\frac{\partial\hat\psi}{\partial x} - \frac{1}{T}\left(\frac{1}{c}\frac{\partial T}{\partial t} + \mu\frac{\partial T}{\partial x}\right)\frac{\partial(u\hat\psi)}{\partial u} = \sigma_a(\hat B - \hat\psi)$$
  with upwinding in $u$ set by the sign of the coefficient, $\hat B\propto T^4$, and the correct material equation. The $T$-factor error in Eq. (42) of the write-up is not in the code.

## Bugs fixed on this branch

### 1. First sweep of each time step uses the previous step's data (MG and VMG)

**Where:** `Base.converge()` calls `rhsUpdate()` before `radiationSweep()`, but `psi_old` and `timeTerm` are advanced in `startTimeStep()`, which runs inside `radiationSweep()`.

**Effect:** The first sweep of each step builds $Q^*$ from $\psi^{n-1}$. That sweep can reproduce $\psi^n$ almost exactly and pass the convergence check, while the material temperature still updates. Radiation does not advance but the material does, so energy is not conserved. In the test problem, VMG lost 0.13% of the total energy in a single step this way, and its error vs. the reference stopped decreasing past 64 groups.

**Fix:** `rhsUpdate()` now calls `startTimeStep()` first, in both `CoupledEquations` and `MovingMeshEquations`.

### 2. MG convergence check never compares successive iterates

**Where:** `CoupledEquations.startTimeStep()` (and the uncoupled `Equations` class used for Reed's problem) never updated `grid.fullTensOld` between iterations.

**Effect:** The quantity checked against the tolerance was $|\psi^{k} - \psi^{n}|$ (current iterate vs. previous time step) rather than $|\psi^{k}-\psi^{k-1}|$. Iterations ended only through the "stagnation" check or by hitting `maxIters`. This inflated MG run times and made MG vs. VMG timing comparisons unfair.

**Fix:** The same-step branch of `startTimeStep()` now sets `grid.fullTensOld` to the latest iterate (as VMG already did).

### 3. MG right reflective boundary lags one full time step

**Where:** `CoupledEquations.radiationSweep()` reads the reflected incoming intensity from `self.fullTens`, which was only refreshed at the start of each time step.

**Effect:** Intensity entering at the right wall was $\psi^n$ instead of the current iterate. In an infinite-medium test this made the solution non-uniform in space and caused a relative energy error of about $10^{-4}$.

**Fix:** `self.fullTens` is refreshed every iteration in `startTimeStep()`.

### 4. Saved radiation history is off by one step

**Where:** `Base.updateAll(index)` wrote the solution at $t_{index+1}$ into slot `index`, overwriting the initial condition. The temperature for the same step was written to slot `index+1`.

**Effect:** Radiation and material temperature were misaligned by one step in every saved file and plot. The final slot stayed empty, which is why `Run.py` slices with `[:-1]`. With log time stepping the last step is about 6% of the total time, so plots of $T_r$ vs. $T_m$ were visibly shifted.

**Fix:** The solution is now stored in slot `index+1`. `checkEnergyConservation` and `Visualize.plotFinalFlux` were adjusted to match.

> **Data in `dataStash/` was produced before these fixes.** Its radiation history is shifted by one step and it may contain unconverged steps. Comparisons built from it should be regenerated.

## Outstanding issues (not fixed)

### 5. Material update is an unlinearized fixed-point iteration (important)

`materialEquation()` updates $T$ with $\sigma$ and $B$ evaluated at the previous iterate. The linearization $T^4_{n+1}\approx T^4_n + 4T^3_n(T^{n+1}-T^n)$ from the write-up is never used. This is probably why the learning rate, polynomial extrapolation, and stagnation workarounds exist.

The iteration fails for large time steps. In the equilibrium test below ($C_v = 0.01$):

- MG overflows (`FloatingPointError`) once $\Delta t \gtrsim 0.2$–$0.3$ ns.
- VMG keeps running at $\Delta t\approx 0.3$ ns but finishes at $T = 0.147$ keV instead of $0.436$ keV, **with no warning**.

**Recommended fix:** a linearized (Fleck–Cummings / effective-scattering) material update, or a Newton iteration on $T$.

### 6. VMG Planckian boundary uses the initial temperature's $u$-coordinate

`MovingMeshEquations.boundaryPlanck()` → `planckVCM()` builds the incoming spectrum assuming $u = \nu/T_{\text{initial}}$. The boundary cell's coordinate is actually $u=\nu/T_{\text{cell}}(t)$. The incoming spectrum in the cell's coordinate should be

$$\hat B_{\text{inc}}(u) = T_{\text{cell}}\,B(uT_{\text{cell}}, T_b),$$

integrated over each $u$-group. This is correct only until the boundary cell's temperature changes, so it affects Marshak-wave runs.

### 7. "Delta" boundary condition has inconsistent units (MG and VMG)

`boundaryPlanck()` / `setBoundaryValues()` compute `totalFlux = sum(du * Bbar_g)` and then set group $g$ to `totalFlux / du[g]`. The tensor stores *group-integrated* intensity, so the delta group should simply hold $\sum_g \bar B_g$. Because `du` is in $\nu$ units for MG and in $u$ units for VMG, the two methods inject different energies.

### 8. Group Planck integrals use one 4-point Simpson panel per group

`planckBar`, `planckg`, `planckBarNumbaVariable`, and the closed-form `sigma_a` all integrate each group with a single 3/8-rule panel. With few log-spaced groups (e.g., a factor of ~5 in frequency per group at G = 8) this is coarse. `energyInitialCondition()` rescales the initial intensity, which hides the error at $t=0$ only; the equilibrium state still carries it. Exact incomplete-Planck integrals (polylog series) or several Gauss points per group would remove it.

### 9. Default configurations crash

- `InfiniteVariable.Parameters` defaults to `boundaryLeft = boundaryRight = "None"`. `setBoundaryValues()` has no branch for that value and raises `UnboundLocalError`. `Run.py`'s `mainMovingMesh()` hits this as-is.
- `InfiniteMedium.Parameters` does not define `matchNu`, so `Base.Grid` raises `AttributeError` (affects `RunBase.mainInfinite()`).
- Both only work through `RunSet.py`, which sets these values.

### 10. Opacity helper typos (currently unused code paths)

- `sigmaAP` in `InfiniteMedium.py` and `DiffEqSolve.py` uses `T*(1/2)`, which is $T/2$, not $\sqrt{T}$.
- The variable-basis versions use $T^{7/2}$ (`InfiniteVariable.py`) and $T^{5/2}$ (`DiffEqSolve.py`), which are inconsistent with each other.

### 11. Minor

- VMG computes $\partial_x T$ with a central difference (`np.gradient`) while the spatial sweep is upwinded, so a spectrum that is uniform in $\nu$ is not preserved exactly. This is a first-order error.
- `Base.converge()` skips its first two iterations (`if it >= 2`), so the `checkEnergy` block (`it == 0`) can never run.

## Verification test: `GroupConvergenceTest.py`

**Problem**
- Infinite homogeneous medium: 2 cells, reflective on both sides, S2.
- Temperature-independent opacity $\sigma(\nu) = 10/(1+\nu^3)$ cm$^{-1}$, $C_v = 0.01$.
- Material starts at 0.4 keV. Radiation starts as a dilute 1.0 keV Planckian carrying the energy of a 0.5 keV Planckian.

**Method**
- MG and VMG run through the production solver (`Base.Base` + `Logic`) for G = 8–128 log-spaced groups.
- The reference is a 1500-group stiff ODE solve (`solve_ivp`, BDF, analytic Jacobian), so it is exact in time.
- Run with `python GroupConvergenceTest.py` (about 2.5 minutes in total).

**Diagnostics**
- max $|T_m - T_{\text{ref}}|$ over time.
- Error in the final cumulative spectrum $\int_0^\nu\phi\,d\nu'$, normalized by the total. This compares the methods even though their group edges differ.
- Relative energy drift.
- Cell-to-cell temperature spread (should be zero in an infinite medium).

### Results before the fixes

| Method | G | max\|ΔT\| | spectrum err | energy drift | T spread |
|---|---|---|---|---|---|
| MG | 32 | 2.8e-3 | 4.0e-2 | 1.1e-4 | 2.4e-5 |
| VMG | 32 | 2.0e-3 | 4.9e-2 | 1.4e-3 | 0 |
| MG | 128 | 2.1e-4 | 2.3e-3 | 9.9e-5 | 2.5e-5 |
| VMG | 128 | 5.1e-4 | 8.5e-3 | 1.5e-3 | 0 |

VMG stopped converging beyond 64 groups (bug 1). MG had spatial non-uniformity and energy drift (bug 3).

### Results after the fixes

| G | MG max\|ΔT\| | VMG max\|ΔT\| | max\|T_MG − T_VMG\| | MG spectrum err | VMG spectrum err | energy drift |
|---|---|---|---|---|---|---|
| 8 | 1.5e-2 | 1.5e-2 | 4.4e-4 | 8.6e-2 | 1.1e-1 | ≤ 1e-9 |
| 16 | 8.7e-3 | 8.2e-3 | 6.3e-4 | 1.1e-1 | 1.1e-1 | ≤ 3e-8 |
| 32 | 2.8e-3 | 2.5e-3 | 3.5e-4 | 4.0e-2 | 4.6e-2 | ≤ 5e-9 |
| 64 | 7.8e-4 | 6.9e-4 | 1.3e-4 | 1.1e-2 | 1.6e-2 | ≤ 8e-9 |
| 128 | 2.0e-4 | 2.2e-4 | 5.3e-5 | 2.8e-3 | 5.8e-3 | ≤ 7e-9 |

- Both methods converge to the reference at roughly second order in temperature, and toward each other.
- VMG spectra converge only at first order, as expected from first-order upwinding of the transfer between $u$-groups.
- Energy is conserved to the iteration tolerance, and the infinite medium stays uniform.

![Group convergence](figures/GroupConvergence.png)

### Relaxation to equilibrium (128 groups)

`equilibrium_plot()` runs both methods to 50 ns: log-spaced steps up to 1 ns, then $\Delta t = 0.1$ ns. The step size is capped because of issue 5.

- Both methods match the reference for material and radiation temperature throughout.
- The material first cools slightly, while absorbing the dilute but spectrally hot radiation, then heats.
- Both reach the equilibrium temperature $T_{eq} = 0.43612$ keV to within about $10^{-6}$ keV.
- Full equilibration takes about 50 ns because the high-frequency tail is only weakly absorbed ($\sigma\sim10^{-3}$ cm$^{-1}$).

![Equilibrium, 128 groups](figures/Equilibrium_128groups.png)
