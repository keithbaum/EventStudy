import numpy as np
import pandas as pd
from statistics import Statistics
from study import runStudy, STATISTICS_ORDER
from events import eventTypeIIParameters

import matplotlib.style
matplotlib.use("Qt5Agg")
matplotlib.style.use('classic')
import matplotlib.pyplot as plt

def calculateErrorTypeI( scenarios, data, index, rejectAlpha):
    results = []
    for numberOfEvents in scenarios:
        statistics = runStudy(data, index, numberOfEvents)
        errorsTypeI = { statisticType:Statistics.errorTypeI(statistic, rejectAlpha, statisticType) for statisticType,statistic in statistics.items() }
        results.append(errorsTypeI)

    return results


def calculateErrorTypeII( scenarios, data, index, rejectAlpha, errorTypeIIParameters):
     results = []
     for numberOfEvents in scenarios:
         result={}
         for eta in errorTypeIIParameters['etas']:
             for Lambda in errorTypeIIParameters['lambdas']:
                 statistics = runStudy(data, index, numberOfEvents, eventTypeIIParameters(eta,Lambda))
                 #This needs to be fixed because statistics is now going to be multidimensional
                 errorsTypeII = [Statistics.errorTypeII(statistic, rejectAlpha, statisticType) for statisticType,statistic in statistics.items()]
                 result[(eta,Lambda)]= errorsTypeII
         results.append(result)
     return results


def resultsDictAsDF(resultsDict):
    return pd.DataFrame(columns=['eta,lambda','T1','T2','Rank','Sign'],data=[ [k]+v for k,v in resultsDict.items()])

def plotResults(scenarios, results):
    color = iter(plt.get_cmap('rainbow')(np.linspace(0, 1, len(STATISTICS_ORDER))))
    for i, error in enumerate(STATISTICS_ORDER):
        statistic = [ result[error] for result in results ]
        c=next(color)
        plt.scatter(scenarios, statistic, label=error, color=c)
        plt.plot(scenarios, statistic, color=c)
    plt.legend()
    plt.show()