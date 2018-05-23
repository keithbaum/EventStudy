from scipy.optimize import curve_fit
import numpy as np
from numba import jit

@jit
def _linear(x,alpha,beta):
    return alpha + beta * x

def performLinearRegressions(datasets, index, estimationWindowSize):
    regressors = np.empty((len(datasets),len(datasets[0].columns),2))
    for i,iteration in enumerate(datasets):
        x = index.loc[ iteration.index ][:estimationWindowSize].values[:,0]
        for assetNumber in range(len(datasets[0].columns)):
            # OLS result params are [alpha beta]
            regressors[i,assetNumber,:] = curve_fit(_linear, x, iteration.iloc[:,assetNumber][:estimationWindowSize].values)[0]

    return regressors


def calculateAbnormalReturns(datasets, marketIndex, eventWindowSize, regressors):
    abnormalReturns = np.empty((len(datasets),len(datasets[0].columns),eventWindowSize))
    for iterationNumber, iteration in enumerate( regressors ):
        vectorIndexes = datasets[iterationNumber].index[-eventWindowSize:]
        marketIndexValues = marketIndex.loc[ vectorIndexes ].iloc[:, 0].values
        currentIteration = datasets[iterationNumber]
        for assetNumber, coefficients in enumerate( iteration ):
            eventWindow = currentIteration.iloc[-eventWindowSize:,assetNumber].values
            abnormalReturn = eventWindow - ( coefficients[0] + coefficients[1]*marketIndexValues )
            abnormalReturns[iterationNumber,assetNumber,:]= abnormalReturn

    return abnormalReturns
