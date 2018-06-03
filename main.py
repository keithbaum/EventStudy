from dataLoader import loadVariables
from errors import calculateErrorTypeI, calculateErrorTypeII, plotResults, resultsDictAsDF

data, index = loadVariables()


scenarios = [50,100,150,200]
errorTypeIIParameters = {'lambdas':[ 0.1, 1, 10 ], 'etas':[ 0.5, 1, 2 ]}
rejectAlpha = 0.05

results = calculateErrorTypeI(scenarios, data, index, rejectAlpha)
plotResults(scenarios, results)

resultsDict = calculateErrorTypeII( [100], data, index, rejectAlpha, errorTypeIIParameters )
print( resultsDictAsDF(resultsDict).to_string() )





