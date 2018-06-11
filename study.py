from datasetPicker import datasetPick, preProcessMarketIndex
from regression import performLinearRegressions, calculateAbnormalReturns
from statistics import Statistics
from windows import generateSimpleWindow
from events import addEvent

STATISTICS_ORDER = ['T1', 'T2', 'rank', 'sign']

def runStudy( data, marketIndex, numberOfAssets=100, errorTypeIIParameters=None ):

    print("Running study for %s assets" % numberOfAssets)
    numberOfSamples = 260
    numberOfIterations = 1000
    estimationWindowSize = 250
    estimationWindow = generateSimpleWindow(0,estimationWindowSize-1,numberOfSamples)
    eventWindow = generateSimpleWindow(estimationWindowSize,numberOfSamples, numberOfSamples)

    datasets, samplesToPickFromIndex = datasetPick(data, numberOfSamples, numberOfAssets, numberOfIterations) #1000 x 100 x 260
    marketIndex = preProcessMarketIndex( samplesToPickFromIndex, marketIndex )
    regressors = performLinearRegressions(datasets, marketIndex, estimationWindow) #1000 x 100 x 2
    abnormalReturns = calculateAbnormalReturns(datasets, marketIndex, (estimationWindow + eventWindow), regressors)
    if errorTypeIIParameters:
        datasets=addEvent(datasets, abnormalReturns, errorTypeIIParameters, eventWindow, estimationWindow)
        #Recalculate abnormal returns because event has been added
        abnormalReturns = calculateAbnormalReturns(datasets, marketIndex, (estimationWindow + eventWindow), regressors)
    statisticsDict = calculateStatistics(datasets, abnormalReturns, estimationWindow, eventWindow)
    return statisticsDict


def calculateStatistics(datasets, abnormalReturns, estimationWindow, eventWindow):
    statistics = Statistics(datasets, abnormalReturns, estimationWindow, eventWindow)
    return { statistic : getattr(statistics, statistic+'_statistic')() for statistic in STATISTICS_ORDER }