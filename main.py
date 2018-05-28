from dataLoader import getReturnsDataframe
from datasetPicker import pickDatasets
from regression import performLinearRegressions
from statistics import Statistics
from windows import generateSimpleWindow

excelPath = '../TP1MNP_PreciosCierre.xlsx'
sheetname= 'Precios'
rawPickledDataPath = 'datos'
transformedPickledDataPath = 'datosTransformadosARetornos'

indexExcelPath = '../spx.xlsx'
indexSheetname = 'Sheet1'
rawIndexPickledDataPath = 'indice'
transformedIndexPickledDataPath = 'indiceTransformadoARetornos'

numberOfSamples = 260
numberOfAssets = 100
numberOfIterations = 1000
estimationWindowSize = 250
eventWindowSize = numberOfSamples - estimationWindowSize
estimationWindow = generateSimpleWindow(0,estimationWindowSize-1,numberOfSamples)
eventWindow = generateSimpleWindow(estimationWindowSize,numberOfSamples, numberOfSamples)


data = getReturnsDataframe(excelPath, sheetname, rawPickledDataPath, transformedPickledDataPath)
index = getReturnsDataframe(indexExcelPath, indexSheetname, rawIndexPickledDataPath, transformedIndexPickledDataPath)

#FOR DEBUG
#import pickle
#import numpy as np
#pkl_file = open('datasets.pkl', 'rb')
#datasets = pickle.load(pkl_file)
#pkl_file.close()
#regressors = np.load('regressors.npy')

#FOR REAL
datasets = pickDatasets(data, numberOfSamples, numberOfAssets, numberOfIterations)
regressors = performLinearRegressions(datasets, index, estimationWindow) #1000 x 100 x 2

statistics = Statistics( datasets, regressors, index, estimationWindow, eventWindow )
T1=statistics.T1_statistic()
T2=statistics.T2_statistic()

from matplotlib import pyplot as plt
plt.plot(T1)
plt.figure()
plt.plot(T2)
plt.show()

statistics.describe( T1, 'T1')
statistics.describe( T2, 'T2')

print("Listo")

