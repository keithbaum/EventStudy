from dataLoader import loadVariables
from errors import calculateErrorTypeI, calculateErrorTypeII, plotResults, resultsDictAsDF

data, index = loadVariables()


scenarios = (5,10,15,20,50,100,150,200,300)
errorTypeIIParameters = {'lambdas':[ 0.1, 1, 10 ], 'etas':[ 0.5, 1, 2 ]}
rejectAlpha = 0.05

results = calculateErrorTypeI(scenarios, data, index, rejectAlpha)
plotResults(scenarios, results)

results = calculateErrorTypeII( scenarios, data, index, rejectAlpha, errorTypeIIParameters )
for i,scenario in enumerate(scenarios):
    print('*'*10+'%s samples'%str(scenario)+'*'*10)
    print( resultsDictAsDF(results[i]).to_string() )
    #ToDo: Plot different power tests grouping by lambda and eta