from dataLoader import getReturnsDataframe
from statistics import Statistics
from study import runStudy

import matplotlib
import matplotlib.style
matplotlib.use("Qt5Agg")
matplotlib.style.use('classic')
import matplotlib.pyplot as plt


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

T1ErrorTypeIPoints= []
T2ErrorTypeIPoints= []
scenarios = [20,30,50,80,100,130,150,200]
rejectAlpha=0.05
for numberOfEvents in scenarios:
    T1,T2 = runStudy( data, index, numberOfEvents )

    T1ErrorTypeI = Statistics.errorTypeI( T1, rejectAlpha )
    T2ErrorTypeI = Statistics.errorTypeI( T2, rejectAlpha )
    T1ErrorTypeIPoints.append(T1ErrorTypeI)
    T2ErrorTypeIPoints.append(T2ErrorTypeI)

plt.scatter(scenarios, T1ErrorTypeIPoints, c='blue', label='T1')
plt.scatter(scenarios, T2ErrorTypeIPoints, c='green', label='T2')
plt.legend()
plt.show()