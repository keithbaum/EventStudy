from scipy.optimize import curve_fit
import numpy as np
from numba import jit

@jit
def _linear(x,alpha,beta):
    return alpha + beta * x

def performLinearRegressions(datasets, marketIndex, estimationWindow):
    regressors = np.empty((datasets.shape[0],datasets.shape[1],2))
    for iteration in range(datasets.shape[0]):
        y = datasets[ iteration, :, estimationWindow.nonzero()[0] ]
        x = marketIndex[ iteration, :, estimationWindow.nonzero()[0] ]
        for assetNumber in range(datasets.shape[1]):
            # OLS result params are [alpha beta]
            regressors[iteration,assetNumber,:] = curve_fit(_linear, x[ :,assetNumber ], y[ :,assetNumber ])[0]

    return regressors

def calculateAbnormalReturns(datasets, marketIndex, window, regressors):
    numberOfAssets = datasets.shape[1]
    numberOfIterations = datasets.shape[0]
    nonzeroWindowSize = len( window.nonzero()[0] )

    abnormalReturns = np.empty((numberOfIterations,numberOfAssets,nonzeroWindowSize))
    for iterationCount in range( regressors.shape[0] ):
        currentIteration = datasets[iterationCount,:,:]
        for assetNumber in range( currentIteration.shape[0] ):
             windowValues = currentIteration[assetNumber,window.nonzero()[0]]
             abnormalReturn = windowValues - ( regressors[iterationCount,assetNumber,0] + regressors[iterationCount,assetNumber,1]*marketIndex[iterationCount,assetNumber,window.nonzero()[0]] )
             abnormalReturns[iterationCount,assetNumber,:]= abnormalReturn

    return abnormalReturns