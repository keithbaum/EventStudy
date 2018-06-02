from dataLoader import loadVariables
from errors import calculateErrorTypeI, calculateErrorTypeII
from errors import plotResults

data, index = loadVariables()


scenarios = [20,30,50,80,100,130,150,200]
errorTypeIIParameters = {'lambdas':[ 0.1, 1, 10 ], 'etas':[ 0.5, 1, 2 ]}
#scenarios = [20]
#errorTypeIIParameters = {'lambdas':[ 0.1 ], 'etas':[ 0.5, 1]}
rejectAlpha = 0.05

results = calculateErrorTypeI(scenarios, data, index, rejectAlpha)
plotResults(scenarios, results)

resultsDict = calculateErrorTypeII( [100], data, index, rejectAlpha, errorTypeIIParameters )
for (eta,Lambda), error in resultsDict.items():
    print( "Error Tipo II para (eta,lambda)=(%s,%s)= %s" % (eta, Lambda, error) )





