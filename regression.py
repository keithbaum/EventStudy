from scipy.optimize import curve_fit
import numpy as np
from numba import jit
from functools import lru_cache
from windows import getStartAndEndOfWindow

@jit
def _linear(x,alpha,beta):
    return alpha + beta * x

def performLinearRegressions(datasets, index, estimationWindow):
    regressors = np.empty((len(datasets),len(datasets[0].columns),2))
    startOfWindow,endOfWindow = getStartAndEndOfWindow(estimationWindow)
    for i,iteration in enumerate(datasets):
        x = index.loc[ iteration.index ][startOfWindow:endOfWindow+1].values[:,0]
        for assetNumber in range(len(datasets[0].columns)):
            # OLS result params are [alpha beta]
            regressors[i,assetNumber,:] = curve_fit(_linear, x, iteration.iloc[:,assetNumber][startOfWindow:endOfWindow+1].values)[0]

    return regressors

def calculateAbnormalReturns(datasets, marketIndex, eventWindow, regressors):

    if not datasets:
        return []
    sampleSize = len(datasets[0])
    numberOfAssets = len( datasets[0].columns) 
    if sampleSize != len(eventWindow):
        return []

    numberOfIterations = len(datasets)
    startOfWindow,endOfWindow = getStartAndEndOfWindow(eventWindow)
    nonzeroEventWindowSize = endOfWindow - startOfWindow + 1

    abnormalReturns = np.empty((numberOfIterations,numberOfAssets,nonzeroEventWindowSize))
    for iterationCount, iteration in enumerate( regressors ):
        vectorIndexes = datasets[iterationCount].index[startOfWindow:endOfWindow+1]
        marketIndexValues = marketIndex.loc[ vectorIndexes ].iloc[:, 0].values
        currentIteration = datasets[iterationCount]
        for assetNumber, coefficients in enumerate( iteration ):
            eventWindowValues = ( currentIteration.iloc[:,assetNumber].values*eventWindow )[startOfWindow:endOfWindow+1]
            abnormalReturn = eventWindowValues - ( coefficients[0] + coefficients[1]*marketIndexValues )
            abnormalReturns[iterationCount,assetNumber,:]= abnormalReturn

    return abnormalReturns