def t1_statistic( datasets, abnormalReturns ):
    '''Asset cross-section mean excess return'''
    ( iterations, numberOfAssets, eventWindowSize ) = abnormalReturns.shape
    for iteration in range(iterations):
        for t in eventWindowSize:
            #ToDo
