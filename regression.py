from scipy.optimize import curve_fit
import numpy as np
from numba import jit
from functools import lru_cache
from windows import getStartAndEndOfWindow

@jit
def _linear(x,alpha,beta):
    return alpha + beta * x

def performLinearRegressions(datasets, marketIndex, estimationWindow):
    regressors = np.empty((len(datasets),len(datasets[0].columns),2))
    for i,iteration in enumerate(datasets):
        y = iteration.values[ estimationWindow.nonzero() ]
        x = marketIndex[estimationWindow.nonzero()]
        for assetNumber in range(len(datasets[0].columns)):
            # OLS result params are [alpha beta]
            regressors[i,assetNumber,:] = curve_fit(_linear, x, y[ :,assetNumber ])[0]

    return regressors

def calculateAbnormalReturns(datasets, marketIndex, window, regressors):
    numberOfAssets = len( datasets[0].columns) 
    numberOfIterations = len(datasets)
    nonzeroWindowSize = len( window.nonzero()[0] )

    abnormalReturns = np.empty((numberOfIterations,numberOfAssets,nonzeroWindowSize))
    for iterationCount, iteration in enumerate( regressors ):
        currentIteration = datasets[iterationCount].values
        for assetNumber, coefficients in enumerate( iteration ):
            windowValues = currentIteration[window.nonzero(),assetNumber]
            abnormalReturn = windowValues - ( coefficients[0] + coefficients[1]*marketIndex[window.nonzero()] )
            abnormalReturns[iterationCount,assetNumber,:]= abnormalReturn

    return abnormalReturns