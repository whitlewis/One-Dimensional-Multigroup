import tkinter as tk
from tkinter import filedialog
import h5py
import Visualize as Vis
import Base as Base
from Base import Constants as const
import os

def loadResults(filepaths=None):
    """Loads datasets and grid/solver metadata from a saved HDF5 results file."""
    # If no path is provided, launch a file dialog to select the file visually
    if filepaths is None:
        root = tk.Tk()
        root.withdraw()
        filepaths = filedialog.askopenfilename(
            title="Select HDF5 Results File",
            initialdir="dataStash",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")],
        )
        if not filepaths:
            print("Load canceled: No file selected.")
            return None, None
    
    all_data = []
    all_params = []
    fileSet = []
    folderSet = []

    for filepath in filepaths:

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
                "runLabel" : f.attrs["runLabel"]
            }
            all_data.append(data)
            all_params.append(params)
            fileName = os.path.basename(filepath).split("_")[0]
            folder = os.path.dirname(filepath)
            folderSet.append(folder)
            fileSet.append(fileName)
            print(f"Successfully loaded: {filepath}")

    return all_data, params, fileSet, folderSet

def plotSet():
    data, params, fileSet, folderSet = loadResults()
    Vis.plotTemperatureLoaded(data, params, fileSet, folderSet, const=const())
    if len(data) < 2:
        Vis.plotTemperatureTimeLoaded(data, params, fileSet, folderSet)
    plotSet = [0, params["nSteps"]//4, params["nSteps"]//2, params["nSteps"]-1]
    freqGroups = 0.5 * (params["freqGrid"][:-1] + params["freqGrid"][1:])
    Vis.plot_spectra_at_times(data["fullTensorPhi"], plotSet, params, fileSet, folderSet, 20, params["maxFreq"], freqs=freqGroups)
    Vis.analyzeRank(data, data["timeSet"], params, fileSet, folderSet, time_indices=None, energy_threshold=[.999, .99, .98 ,.95], tol=None, plot=True)
    # Vis.analyzeRank(data["fullTensorPhi"], data["timeSet"], energy_threshold=None, tol = [1e-8, 1e-14, 1e-16, 1e-18], plot = True)
    Vis.analyzeRank(data, data["timeSet"], params, fileSet, folderSet, time_indices=None, energy_threshold=None, tol=None, plot=True)

plotSet()