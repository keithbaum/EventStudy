import numpy as np

def datasetPick( data, numberOfSamples, numberOfAssets, numberOfIterations ):
    totalAssets = data.shape[0]
    totalSamples = data.shape[1]

    assetPickerArray = assetPicker(numberOfIterations, numberOfAssets, totalAssets)
    samplePickerArray = samplePicker(numberOfIterations, numberOfAssets, numberOfSamples, totalSamples)

    values = np.empty((numberOfIterations, numberOfAssets, numberOfSamples))
    for iteration in range(numberOfIterations):
        for assetNumber in range(numberOfAssets):
            values[iteration,assetNumber,:]= data[ assetPickerArray[iteration,assetNumber], samplePickerArray[iteration,assetNumber] ]

    return ( values, samplePickerArray)


def assetPicker(numberOfIterations, numberOfAssets, totalAssets):
    assetPicker = np.empty((numberOfIterations, numberOfAssets), dtype=int)
    for i in range(numberOfIterations):
        assetPicker[i, :] = np.random.choice(range(totalAssets), numberOfAssets, replace=False)

    return assetPicker

def samplePicker(numberOfIterations, numberOfAssets, numberOfSamples, totalSamples):
    samplePickerArray = np.empty((numberOfIterations, numberOfAssets, numberOfSamples), dtype=int)
    for iteration in range(numberOfIterations):
        for assetNumber in range(numberOfAssets):
            samplePickerArray[iteration, assetNumber,:] = singleSamplePicker(numberOfSamples, totalSamples)

    return samplePickerArray

def singleSamplePicker(numberOfSamples, totalSamples):
    highestStartSample = totalSamples - numberOfSamples
    start = np.random.randint( 0, highestStartSample, 1, dtype=int)
    samplePicker = np.arange(start, start+numberOfSamples)

    return samplePicker

def preProcessMarketIndex( samplesToPickFromIndex, marketIndex ):
    result = np.empty( samplesToPickFromIndex.shape )
    for iteration in range( samplesToPickFromIndex.shape[0] ):
        for numberOfAsset in range(samplesToPickFromIndex.shape[1]):
            result[iteration,numberOfAsset,:] = marketIndex[samplesToPickFromIndex[iteration, numberOfAsset,:]]
    return result

def DFlistToArray(datasets):
    if isinstance(datasets, np.ndarray):
        return datasets

    values = np.empty((len(datasets),len(datasets[0].columns),len(datasets[0].index)))
    for i,iteration in enumerate(datasets):
        values[i,:,:] = iteration.values.T
    return values