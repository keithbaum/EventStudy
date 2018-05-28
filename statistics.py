import numpy as np
from regression import calculateAbnormalReturns
from windows import generateSimpleWindow, getStartAndEndOfWindow

class Statistics( object ):
    def __init__( self, datasets, regressors, marketIndex, estimationWindow, eventWindow):
        self.datasets = datasets
        self.regressors = regressors
        self.marketIndex = marketIndex
        self.estimationWindow = estimationWindow
        self.eventWindow = eventWindow

        self.abnormalReturns = calculateAbnormalReturns( datasets,
                                                        marketIndex,
                                                        estimationWindow,
                                                        regressors)
        t0,_ = getStartAndEndOfWindow(eventWindow)
        self.t0AbnormalReturns = calculateAbnormalReturns( datasets,
                                                        marketIndex,
                                                        generateSimpleWindow(t0,t0,len(eventWindow)),
                                                        regressors)

    def T1_statistic( self ):
        '''Asset cross-section mean excess return'''
        AHat = self.crossSectionAverageAbnormalReturnOnWindow( self.abnormalReturns )
        AHatSquared = np.power(AHat,2)
        S = np.sqrt( np.mean( AHatSquared, axis=1 ) )
        A0 = self.crossSectionAverageAbnormalReturnOnWindow( self.t0AbnormalReturns )

        return np.squeeze(A0)/S

    def T2_statistic( self ):
        '''Mean standarized excess return'''
        S = np.sqrt( self.crossTimeAverageSquaredAbnormalReturnOnWindow( self.abnormalReturns ) )
        numberOfAssets = len( self.datasets[0].columns ) 
        A0 = self.t0AbnormalReturns
        return np.mean( np.squeeze(A0)/S, axis=1 )*np.sqrt(numberOfAssets)

    @staticmethod
    def crossSectionAverageAbnormalReturnOnWindow( abnormalReturns ):
        crossSectionAverage = np.mean( abnormalReturns, axis=1 )

        return crossSectionAverage

    @staticmethod
    def crossTimeAverageSquaredAbnormalReturnOnWindow( abnormalReturns ):
        crossTimeAverage = np.mean( np.power( abnormalReturns, 2 ), axis=2 )

        return crossTimeAverage
    

        


