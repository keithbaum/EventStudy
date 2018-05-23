from dataLoader import getReturnsDataframe
from datasetPicker import pickDatasets
from regression import performLinearRegressions, calculateAbnormalReturns
from t_statistics import t1_statistic

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

data = getReturnsDataframe(excelPath, sheetname, rawPickledDataPath, transformedPickledDataPath)
index = getReturnsDataframe(indexExcelPath, indexSheetname, rawIndexPickledDataPath, transformedIndexPickledDataPath)
datasets = pickDatasets(data, numberOfSamples, numberOfAssets, numberOfIterations)

regressors = performLinearRegressions(datasets, index, estimationWindowSize) #1000 x 100 x 2
abnormalReturns = calculateAbnormalReturns( datasets, index, eventWindowSize, regressors) #1000 x 100 x 10

print("Listo")

