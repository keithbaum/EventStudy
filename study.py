from datasetPicker import pickDatasets, preProcessMarketIndex
from regression import performLinearRegressions
from statistics import Statistics
from windows import generateSimpleWindow

def runStudy( data, marketIndex, numberOfAssets=100 ):

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
    marketIndex = preProcessMarketIndex( datasets, marketIndex )
    regressors = performLinearRegressions(datasets, marketIndex, estimationWindow) #1000 x 100 x 2

    statistics = Statistics( datasets, regressors, marketIndex, estimationWindow, eventWindow )
    T1=statistics.T1_statistic()
    T2=statistics.T2_statistic()
    rankStatistic = statistics.Rank_statistic()
    signStatistic = statistics.Sign_statistic()

    return (T1,T2, rankStatistic, signStatistic)