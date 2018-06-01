from dataLoader import getReturnsDataframe
from statistics import Statistics
from study import runStudy

import matplotlib
import matplotlib.style
matplotlib.use("Qt5Agg")
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np


def loadVariables():
    excelPath = '../TP1MNP_PreciosCierre.xlsx'
    sheetname= 'Precios'
    rawPickledDataPath = 'datos'
    transformedPickledDataPath = 'datosTransformadosARetornos'

    indexExcelPath = '../spx.xlsx'
    indexSheetname = 'Sheet1'
    rawIndexPickledDataPath = 'indice'
    transformedIndexPickledDataPath = 'indiceTransformadoARetornos'

    data = getReturnsDataframe(excelPath, sheetname, rawPickledDataPath, transformedPickledDataPath)
    index = getReturnsDataframe(indexExcelPath, indexSheetname, rawIndexPickledDataPath, transformedIndexPickledDataPath)
    return (data, index)


def calculateErrorTypeI( scenarios, data, index, rejectAlpha):
    results = []
    for numberOfEvents in scenarios:
        statistics = runStudy(data, index, numberOfEvents)
        errorsTypeI = [Statistics.errorTypeI(statistic, rejectAlpha) for statistic in statistics]
        results.append(errorsTypeI)

    results = np.array(results).T.tolist()
    return results

def calculateErrorTypeII( scenarios, data, index, rejectAlpha, errorTypeIIParameters):
     results = []
     for numberOfEvents in scenarios:
         statistics = runStudy(data, index, numberOfEvents, errorTypeIIParameters)
         #This needs to be fixed because statistics is now going to be multidimensional
         errorsTypeII = [ Statistics.errorTypeII(statistic, rejectAlpha) for statistic in statistics ]
         results.append(errorsTypeII)

     results = np.array(results).T.tolist()
     return results

data, index = loadVariables()


scenarios = [20,30,50,80,100]#,130,150,200]
rejectAlpha = 0.05
errorTypeIIParameters = {'lambdas':[ 0.1, 1, 10 ], 'etas':[ 0.5, 1, 2 ]}


results = calculateErrorTypeI( scenarios, data, index, rejectAlpha )
color = iter(cm.rainbow(np.linspace(0, 1, len(results))))
for i, error in enumerate(results):
    plt.scatter(scenarios, error, label="T%s" % str(i + 1), c=next(color))
plt.legend()
plt.show()

# results = calculateErrorTypeII( [100], data, index, rejectAlpha, errorTypeIIParameters )



