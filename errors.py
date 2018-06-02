import numpy as np
from statistics import Statistics
from study import runStudy
from events import eventTypeIIParameters

import matplotlib.style
matplotlib.use("Qt5Agg")
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def calculateErrorTypeI( scenarios, data, index, rejectAlpha):
    results = []
    for numberOfEvents in scenarios:
        statistics = runStudy(data, index, numberOfEvents)
        errorsTypeI = [Statistics.errorTypeI(statistic, rejectAlpha) for statistic in statistics]
        results.append(errorsTypeI)

    results = np.array(results).T.tolist()
    return results


def calculateErrorTypeII( scenarios, data, index, rejectAlpha, errorTypeIIParameters):
     results = {}
     for eta in errorTypeIIParameters['etas']:
         for Lambda in errorTypeIIParameters['lambdas']:
             for numberOfEvents in scenarios:
                 statistics = runStudy(data, index, numberOfEvents, eventTypeIIParameters(eta,Lambda))
                 #This needs to be fixed because statistics is now going to be multidimensional
                 errorsTypeII = [ Statistics.errorTypeII(statistic, rejectAlpha) for statistic in statistics ]
                 results[(eta,Lambda)]= errorsTypeII
     return results

def plotResults(scenarios, results):
    color = iter(cm.rainbow(np.linspace(0, 1, len(results))))
    for i, error in enumerate(results):
        plt.scatter(scenarios, error, label="T%s" % str(i + 1), c=next(color))
    plt.legend()
    plt.show()