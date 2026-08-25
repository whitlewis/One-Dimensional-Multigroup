import tkinter as tk
from tkinter import filedialog
import h5py
import Visualize as Vis
import Base as Base
from Base import Constants as const

def loadResults(filepath=None):
    """Loads datasets and grid/solver metadata from a saved HDF5 results file."""
    # If no path is provided, launch a file dialog to select the file visually
    if filepath is None:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select HDF5 Results File",
            initialdir="dataStash",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")],
        )
        if not filepath:
            print("Load canceled: No file selected.")
            return None, None

    with h5py.File(filepath, "r") as f:
        # Load numerical arrays
        data = {
            "fullTensorPhi": f["fullTensorPhi"][:],
            "temperatureSet": f["temperatureSet"][:],
            "timeSet": f["timeSet"][:],
        }

        # Load grid and solver parameters stored in attributes
        params = {
            "spaceGrid": f.attrs["spaceGrid"],
            "spaceMid": f.attrs["spaceMid"],
            "freqGrid": f.attrs["freqGrid"],
            "dt": f.attrs["dt"],
            "dx": f.attrs["dx"],
            "nBins": f.attrs["nBins"],
            "nSteps": f.attrs["nSteps"],
            "sn": f.attrs["sn"],
            "maxFreq": f.attrs["maxFreq"],
        }

    print(f"Successfully loaded: {filepath}")
    return data, params

def plotSet():
    data, params = loadResults()
    Vis.plotTemperatureLoaded(data, params, const=const())
    Vis.plotTemperatureTimeLoaded(data, params)
    plotSet = [0, params["nSteps"]//4, params["nSteps"]//2, params["nSteps"]-1]
    freqGroups = 0.5 * (params["freqGrid"][:-1] + params["freqGrid"][1:])
    Vis.plot_spectra_at_times(data["fullTensorPhi"], plotSet, 20, params["maxFreq"], freqs=freqGroups)
    Vis.analyzeRank(data["fullTensorPhi"], data["timeSet"], time_indices=None, energy_threshold=[.999999999, .999999, .9999  ,.99], tol=None, plot=True)
    Vis.analyzeRank(data["fullTensorPhi"], data["timeSet"], energy_threshold=None, tol = [1e-8, 1e-14, 1e-16, 1e-18], plot = True)
    Vis.analyzeRank(data["fullTensorPhi"], data["timeSet"], time_indices=None, energy_threshold=None, tol=None, plot=True)

plotSet()