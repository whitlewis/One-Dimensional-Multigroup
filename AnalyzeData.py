from pdb import main
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import h5py
import Visualize as Vis
import Base as Base
from Base import Constants as const
import os
import numpy as np

def fileLoad():
        root = tk.Tk()
        root.withdraw()
        filepaths = filedialog.askopenfilenames(
            title="Select HDF5 Results File",
            initialdir="dataStash",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")],
        )
        root.destroy()
        return filepaths

def loadResults(filepaths=None):
    """Loads datasets and grid/solver metadata from a saved HDF5 results file."""
    # If no path is provided, launch a file dialog to select the file visually

    if filepaths is None:

        filepaths = fileLoad()
        msgbox = tk.messagebox.askquestion ('Add files','add extra files',icon = 'warning')
        while msgbox.lower() =='yes':
            loadedFile = fileLoad()
            filepaths += loadedFile
            msgbox = tk.messagebox.askquestion ('Add files','add extra files',icon = 'warning')


        if not filepaths:
            print("Load canceled: No file selected.")
            return None, None, None, None
    
    all_data = []
    all_params = []
    fileSet = []
    folderSet = []


    for filepath in filepaths:

        with h5py.File(filepath, "r") as f:
            # Load numerical arrays
            # Load numerical arrays
            data = {
                "fullTensorPhi": f["fullTensorPhi"][:],
                "temperatureSet": f["temperatureSet"][:],
                "timeSet": f["timeSet"][:],
            }

            # Load grid and solver parameters stored in attributes
            spaceGrid = f.attrs.get("spaceGrid", None)
            if spaceGrid is None:
                spaceGrid = f["spaceGrid"][:]

            spaceMid = f.attrs.get("spaceMid", None)
            if spaceMid is None:
                spaceMid = f["spaceMid"][:]

            freqGrid = f.attrs.get("freqGrid", None)
            if freqGrid is None:
                freqGrid = f["freqGrid"][:]

            dt = f.attrs.get("dt", None)
            if dt is None:
                dt = f["dt"][:]

            fileName = os.path.splitext(os.path.basename(filepath))[0]
            folder = os.path.basename(os.path.dirname(filepath))
            nSteps = f.attrs.get("nSteps", None)
            groups = f.attrs.get("groups", None)
            movingCoordinates = bool(
                f.attrs.get("movingCoordinates", False)
            )
            methodName = "VMG" if movingCoordinates else "MG"
            groupLabel = f" ({groups} groups)" if groups is not None else ""
            params = {
            "spaceGrid": spaceGrid,
            "spaceMid": spaceMid,
            "freqGrid": freqGrid,
            "dt": dt,
            "dx": f.attrs.get("dx", None),
            "nBins": f.attrs.get("nBins", None),
            "nSteps": nSteps,
            "sn": f.attrs.get("sn", None),
            "maxFreq": f.attrs.get("maxFreq", None),
            "runLabel": f.attrs.get("runLabel", None),
            "groups": groups,
            "solveType": f.attrs.get("solveType", None),
            "MMSCompare": bool(f.attrs.get("MMSCompare", False)),
            "mmsT0": f.attrs.get("mmsT0", None),
            "mmsSigmaX": f.attrs.get("mmsSigmaX", None),
            "mmsTau": f.attrs.get("mmsTau", None),
            "movingCoordinates": movingCoordinates,
            "plotLabel": f"{methodName}{groupLabel}",
            }
            all_data.append(data)
            all_params.append(params)
            folderSet.append(folder)
            print(folder)
            fileSet.append(fileName)
            print(f"Successfully loaded: {filepath}")

    return all_data, all_params, fileSet, folderSet

def plotSet(filepaths=None):
    data, params, fileSet, folderSet = loadResults(filepaths)
    if data is None:
        return
    mmsResults = [
        (runData, runParams, runParams["plotLabel"])
        for runData, runParams in zip(data, params)
        if runParams["MMSCompare"]
    ]
    if mmsResults:
        Vis.plotMMSComparisonsLoaded(mmsResults)
        Vis.plotTemperatureMMSLoaded(mmsResults, binIndex=20)
    nonMmsResults = [
        (runData, runParams, file, folder)
        for runData, runParams, file, folder in zip(
            data, params, fileSet, folderSet
        )
        if not runParams["MMSCompare"]
    ]
    if nonMmsResults:
        nonMmsData, nonMmsParams, nonMmsFiles, nonMmsFolders = zip(
            *nonMmsResults
        )
        Vis.plotTemperatureLoaded(
            nonMmsData, nonMmsParams, nonMmsFiles, nonMmsFolders,
            const=const(),
        )
        if len(nonMmsData) < 2:
            Vis.plotTemperatureTimeLoaded(
                nonMmsData, nonMmsParams, nonMmsFiles, nonMmsFolders
            )
        # plotSet = [0, int(params[0]["nSteps"])//4, params[0]["nSteps"]//2, params[0]["nSteps"]-1]
        plotIndices = np.floor(
            np.geomspace(1, nonMmsParams[0]["nSteps"] - 1, 4)
        ).astype(int)
        Vis.plot_spectra_at_times(
            nonMmsData, plotIndices, nonMmsParams, nonMmsFiles,
            nonMmsFolders, 20, nonMmsParams[0]["maxFreq"],
        )
    # Vis.analyzeRank(data, data[0]["timeSet"], params, fileSet, folderSet, time_indices=None, energy_threshold=[.999, .99, .98 ,.95], tol=None, plot=True)
    # Vis.analyzeRank(data["fullTensorPhi"], data["timeSet"], energy_threshold=None, tol = [1e-8, 1e-14, 1e-16, 1e-18], plot = True)
    # Vis.analyzeRank(data, data[0]["timeSet"], params, fileSet, folderSet, time_indices=None, energy_threshold=None, tol=None, plot=True)

if __name__ == "__main__":
    plotSet()