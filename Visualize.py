import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from matplotlib.animation import FuncAnimation


def plotOut(phil, grid, angle=0, label=""):
    
    x = 0.5 * (grid.grid[:-1] + grid.grid[1:])  # cell centers

    plt.plot(x, phil[angle], label=f"{label} μ={grid.muSet[angle]:.2f}")
    plt.xlabel("Position x")
    plt.ylabel("Angular Flux ψ")
    plt.legend()
    plt.show()


def PlotTemperature(grid):
    x = grid.spaceMid # cell centers
    labelT = "Final Temperature T"
    T = grid.temperatureSet[:, -1]  # Final temperature distribution at the last time step
    plt.plot(x, T, label=labelT)
    plt.xlabel("Position x")
    plt.ylabel("Temperature T")
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"figures/{labelT}_{timestamp}.pdf"
    # plt.savefig(filename)
    plt.show()

def plotTemperatureTime(grid):
    x = grid.spaceMid  # cell centers
    time = grid.timeSet
    T_time = grid.temperatureSet  # shape: (nBins, nSteps+1)

    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(
        x,
        time,
        T_time.T,
        shading='auto',
        cmap='viridis'
    )

    fig.colorbar(pcm, label='Temperature T', ax=ax)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Time')
    ax.set_title('Space-Time Temperature Distribution')
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)

    plt.tight_layout()
    plt.show()

def plotFinalFlux(grid, label="Final Scalar Flux"):
    x = grid.spaceMid
    phi_by_group = grid.fullTensorPhiTime[-2]  # shape: (nFreq, nBins)

    plt.figure()
    for group, phi in enumerate(phi_by_group):
        plt.plot(x, phi, label=f"Group {group + 1}")

    # if phi_by_group.shape[0] > 1:
    #     plt.plot(x, np.sum(phi_by_group, axis=0), "k--", label="Total")

    plt.xlabel("Position x")
    plt.ylabel("Scalar Flux")
    plt.title(label)
    plt.legend()
    plt.show()


def plotOutScalar(phil, grid, constants, label="Scalar Flux φ"):

    x = 0.5 * (grid.grid[:-1] + grid.grid[1:])
    scalar = np.sum(grid.w[:, None] * phil, axis=0)
    split_idx = np.searchsorted(x, 0)

    x = x[split_idx:]
    sol_half = scalar[split_idx:] # Slice the spatial dimension (axis 1)

    plt.plot(x, sol_half, label=label)
    plt.xlabel("Position x")
    plt.ylabel("Scalar Flux φ")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/{label}_{timestamp}.pdf"
    plt.savefig(filename)
    plt.show()

    #Plot T_r for comparison
    T_r = (sol_half / constants.a / constants.c)**0.25
    plt.plot(x, T_r, label=f"{label} T_r")  
    plt.xlabel("Position x")
    plt.ylabel("Radiation Temperature T_r")
    plt.legend()
    plt.show()

def plotSpaceTime(phi, params, dt):
    plt.figure(figsize=(8, 5))

    plt.imshow(
        phi,
        aspect='auto',
        origin='lower',
        extent=[params.xMin, params.xMax, 0, params.timeMax]
    )

    plt.colorbar(label='Scalar Flux')
    plt.xlabel('Position x')
    plt.ylabel('Time')
    plt.title('Space-Time Scalar Flux')

    plt.show()

def plotSnapshot(sol, grid, title=None, label="Flux"):
    sol = np.asarray(sol)

    x = grid.spaceMid
    plt.figure()
    plt.plot(x, sol)

    plt.xlabel("Position x")
    plt.ylabel(label)

    if title:
        plt.title(title)

    plt.grid(True)
    plt.show()

def animate_solution(fullPhi, params, grid, interval=15):
    sol = np.array(fullPhi)   # shape: (Nt, Nx)
    x_full = np.linspace(params.xMin, params.xMax, sol.shape[1])

    # 1. Find the index where x >= 0
    # np.searchsorted finds the first index where x would be inserted to maintain order
    split_idx = np.searchsorted(x_full, 0)

    # 2. Slice the arrays to only include x >= 0
    x = x_full[split_idx:]
    sol_half = sol[:, split_idx:] # Slice the spatial dimension (axis 1)

    fig, ax = plt.subplots()
    line, = ax.plot(x, sol_half[0])

    # 3. Update plot limits for the new range
    ax.set_xlim(0, params.xMax)
    ax.set_ylim(np.min(sol_half), np.max(sol_half))
    ax.set_xlabel('x')
    ax.set_ylabel('Scalar Flux')

    def update(frame):
        line.set_ydata(sol_half[frame])
        # Using frame * dt if grid.timeSet isn't exactly matched to Nt
        ax.set_title(f'Time: {frame * grid.dt[frame]:.2f}')
        return line,

    anim = FuncAnimation(
        fig,
        update,
        frames=sol_half.shape[0],
        interval=interval
    )
    plt.show()