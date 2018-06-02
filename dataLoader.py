import os

import numpy as np
import pandas as pd


def loadData(pickledDataFile, excelPath, sheetname):
    if not os.path.isfile(pickledDataFile):
        data=pd.read_excel(excelPath, sheetname=sheetname, index_col=0).sort_index()
        data.to_pickle(pickledDataFile)

    return pd.read_pickle(pickledDataFile)


def priceSeriesToReturnSeries(values):
    return np.log(values[1:]/values[:-1])


def pricesDataFrameToReturnsDataFrame( df ):
    result = pd.DataFrame( index= df.index[1:], columns=df.columns )
    for ticker in df:
        result.loc[:,ticker] = priceSeriesToReturnSeries( df[ticker].values )

    return result


def getReturnsDataframe(excelPath, sheetname, rawPath, transformedDataPath):
    if not os.path.isfile(transformedDataPath):
        data = loadData(rawPath, excelPath, sheetname)
        transformed = pricesDataFrameToReturnsDataFrame(data)
        transformed.to_pickle(transformedDataPath)

    return pd.read_pickle(transformedDataPath)


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