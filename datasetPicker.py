import numpy as np
import pandas as pd

def pickDatasets( df, numberOfSamples, numberOfAssets, numberOfIterations ):
    totalAssets = len( df.columns )
    totalSamples = len( df.index )

    assetPicker = assetPickerArray(numberOfIterations, numberOfAssets, totalAssets)
    samplePicker = samplePickerArray(numberOfIterations, numberOfSamples, totalSamples)

    datasets = []
    for i in range(numberOfIterations):
        datasets.append( df.iloc[ samplePicker[i], assetPicker[i] ] )

    return datasets


def assetPickerArray(numberOfIterations, numberOfAssets, totalAssets):
    assetPicker = np.empty((numberOfIterations, numberOfAssets))
    for i in range(numberOfIterations):
        assetPicker[i, :] = np.random.choice(range(totalAssets), numberOfAssets, replace=False)

    return assetPicker

def samplePickerArray(numberOfIterations, numberOfSamples, totalSamples):
    samplePicker = np.empty((numberOfIterations, numberOfSamples))
    highestStartSample = totalSamples - numberOfSamples
    for i in range(numberOfIterations):
        start = np.random.randint( 0, highestStartSample, 1, int)
        samplePicker[i,:] = np.arange(start, start+numberOfSamples)

    return samplePicker

def preProcessMarketIndex( datasets, marketIndex ):
    return np.squeeze( marketIndex.loc[ datasets[0].index ].values )