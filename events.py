from collections import namedtuple
import numpy as np

eventTypeIIParameters = namedtuple('eventTypeIIParameters',['eta','Lambda'])

def addEvent(datasets, abnormalReturns, errorTypeIIParameters, eventWindow, estimationWindow):
    eventWindowSize = np.count_nonzero( eventWindow )
    additionalEventWindows = np.empty((datasets.shape[0], datasets.shape[1], eventWindowSize))
    sigmas = np.std(abnormalReturns[:,:,estimationWindow.nonzero()[0]], axis=2,ddof=1)
    for iterationNumber in range( datasets.shape[0] ):
        for assetNumber in range( datasets.shape[1] ):
            additionalEventWindows[iterationNumber,assetNumber,:] = errorTypeIIParameters.eta*sigmas[iterationNumber,assetNumber]*np.exp(-np.arange(0,eventWindowSize,1)/errorTypeIIParameters.Lambda)
    datasets[:,:,eventWindow.nonzero()[0]]+= additionalEventWindows

    return datasets