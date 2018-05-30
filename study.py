from datasetPicker import pickDatasets
from regression import performLinearRegressions
from statistics import Statistics
from windows import generateSimpleWindow

def runStudy( data, index, numberOfAssets=100 ):

    print("Running study for %s assets" % numberOfAssets)
    numberOfSamples = 260
    numberOfIterations = 1000
    estimationWindowSize = 250
    estimationWindow = generateSimpleWindow(0,estimationWindowSize-1,numberOfSamples)
    eventWindow = generateSimpleWindow(estimationWindowSize,numberOfSamples, numberOfSamples)

    #FOR DEBUG
    #import pickle
    #import numpy as np
    #pkl_file = open('datasets.pkl', 'rb')
    #datasets = pickle.load(pkl_file)
    #pkl_file.close()
    #regressors = np.load('regressors.npy')

    #FOR REAL
    datasets = pickDatasets(data, numberOfSamples, numberOfAssets, numberOfIterations) #1000 x 100 x 260
    regressors = performLinearRegressions(datasets, index, estimationWindow) #1000 x 100 x 2

    statistics = Statistics( datasets, regressors, index, estimationWindow, eventWindow )
    T1=statistics.T1_statistic()
    T2=statistics.T2_statistic()

    return (T1,T2)