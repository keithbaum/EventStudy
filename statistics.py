import numpy as np
import scipy.stats
from regression import calculateAbnormalReturns
from windows import getStartAndEndOfWindow

class Statistics( object ):
    def __init__( self, datasets, regressors, marketIndex, estimationWindow, eventWindow):
        self.datasets = datasets
        self.regressors = regressors
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

    def Rank_statistic( self ):
        abnormalReturns = np.concatenate( ( self.estimationAbnormalReturns, self.eventAbnormalReturns ), axis = 2 )
        rankMatrix = np.argsort( abnormalReturns, axis=2 )
        rankMean = ( rankMatrix.shape[2] - 1 )/2
        S = np.sqrt( np.mean( np.power( np.mean( rankMatrix - rankMean, axis=1 ), 2), axis=1 ) )
        cummulativeRankInEventWindow = np.sum( np.mean( rankMatrix[:,:,self.eventWindow.nonzero()[0]]-rankMean, axis=1 ),axis=1)
        return cummulativeRankInEventWindow/S/np.sqrt(self.eventWindowSize)

    def Sign_statistic(self):
        abnormalReturns = np.concatenate((self.estimationAbnormalReturns, self.eventAbnormalReturns), axis=2)
        signMatrix = np.sign( abnormalReturns )
        absAbnormalReturns = np.abs( abnormalReturns )
        rankMatrix = np.argsort(absAbnormalReturns, axis=2)
        signedRankMatrix = rankMatrix*signMatrix
        return np.sum( signedRankMatrix, axis=2 )/np.sqrt( np.sum( np.power( signedRankMatrix, 2), axis=2 ) )



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

