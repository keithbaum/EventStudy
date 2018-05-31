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


############################################################

data, index = loadVariables()


scenarios = [20,30,50,80,100,130,150,200]
rejectAlpha = 0.05

results = []
for numberOfEvents in scenarios:
    statistics = runStudy( data, index, numberOfEvents )
    errorsTypeI = [ Statistics.errorTypeI( statistic, rejectAlpha ) for statistic in statistics ]
    results.append ( errorsTypeI )

color=iter(cm.rainbow(np.linspace(0,1,len(results))))
for i,errorI in enumerate(results):
    plt.scatter( scenarios, errorI, label ="T%s"%i, c=next(color) )

plt.legend()
plt.show()