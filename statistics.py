import numpy as np
import scipy.stats
from regression import calculateAbnormalReturns
from windows import generateSimpleWindow, getStartAndEndOfWindow

class Statistics( object ):
    def __init__( self, datasets, regressors, marketIndex, estimationWindow, eventWindow):
        self.datasets = datasets
        self.regressors = regressors
        self.marketIndex = marketIndex
        self.estimationWindow = estimationWindow
        self.eventWindow = eventWindow

        start,end = getStartAndEndOfWindow(eventWindow)
        self.eventWindowSize = end-start+1

        self.estimationAbnormalReturns = calculateAbnormalReturns( datasets,
                                                        marketIndex,
                                                        estimationWindow,
                                                        regressors)
        self.eventAbnormalReturns = calculateAbnormalReturns( datasets,
                                                        marketIndex,
                                                        eventWindow,
                                                        regressors)

    def T1_statistic( self ):
        '''Asset cross-section mean excess return'''
        AHat = self.crossSectionAverageAbnormalReturnOnWindow( self.estimationAbnormalReturns )
        AHatSquared = np.power(AHat,2)
        S = np.sqrt( np.mean( AHatSquared, axis=1 ) )
        eventCARs = np.sum( self.crossSectionAverageAbnormalReturnOnWindow( self.eventAbnormalReturns ), axis=1 )

        return np.squeeze(eventCARs)/S/np.sqrt(self.eventWindowSize)

    def T2_statistic( self ):
        '''Mean standarized excess return'''
        S = np.sqrt( self.crossTimeAverageSquaredAbnormalReturnOnWindow( self.estimationAbnormalReturns ) )
        numberOfAssets = len( self.datasets[0].columns ) 
        eventCARs = np.sum( self.eventAbnormalReturns, axis = 2 )
        return np.mean( np.squeeze(eventCARs)/S/np.sqrt(self.eventWindowSize), axis=1 )*np.sqrt(numberOfAssets)

    @staticmethod
    def crossSectionAverageAbnormalReturnOnWindow( abnormalReturns ):
        crossSectionAverage = np.mean( abnormalReturns, axis=1 )

        return crossSectionAverage

    @staticmethod
    def crossTimeAverageSquaredAbnormalReturnOnWindow( abnormalReturns ):
        crossTimeAverage = np.mean( np.power( abnormalReturns, 2 ), axis=2 )

        return crossTimeAverage

    @staticmethod        
    def describe( series, seriesName='' ):
        print( "Estadistico %s" % seriesName )
        print( scipy.stats.describe( series ) )

    @staticmethod
    def errorTypeI( statistic, alpha):
        sigma = np.std( statistic )
        zAlphaBillateral = scipy.stats.norm.ppf( 1-alpha/2 )
        thresholdForRejection = sigma * zAlphaBillateral
        occurences = np.where( np.abs(statistic)>thresholdForRejection )[0]

        return len(occurences)/len(statistic)

