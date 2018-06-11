import numpy as np
import scipy.stats

class Statistics( object ):

    UNILATERAL = lambda x: 1-x
    BILATERAL = lambda x: 1-x/2

    TEST_TYPE = {
                    'T1'    : BILATERAL,
                    'T2'    : BILATERAL,
                    'rank'  : BILATERAL,
                    'sign'  : BILATERAL,
    }

    def __init__( self, datasets, abnormalReturns, estimationWindow, eventWindow):
        self.datasets = datasets
        self.estimationWindow = estimationWindow
        self.eventWindow = eventWindow
        self.eventWindowSize = np.count_nonzero(eventWindow)

        self.abnormalReturns = abnormalReturns
        self.estimationAbnormalReturns = self.abnormalReturns[:,:,self.eventWindow.nonzero()[0]]
        self.eventAbnormalReturns = self.abnormalReturns[:,:,self.estimationWindow.nonzero()[0]]

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
        numberOfAssets = self.datasets.shape[1]
        eventCARs = np.sum( self.eventAbnormalReturns, axis = 2 )
        return np.mean( np.squeeze(eventCARs)/S/np.sqrt(self.eventWindowSize), axis=1 )*np.sqrt(numberOfAssets)

    def rank_statistic( self ):
        rankMatrix = np.argsort( self.abnormalReturns, axis=2 )
        rankMean = ( rankMatrix.shape[2] - 1 )/2
        S = np.sqrt( np.mean( np.power( np.mean( rankMatrix[:,:,self.eventWindow.nonzero()[0]] - rankMean, axis=1 ), 2), axis=1 ) )
        cummulativeRankInEventWindow = np.sum( np.mean( rankMatrix[:,:,self.eventWindow.nonzero()[0]]-rankMean, axis=1 ), axis=1 )
        return cummulativeRankInEventWindow/S/np.sqrt(self.eventWindowSize)

    def sign_statistic(self):
        signMatrix = np.sign( self.eventAbnormalReturns )
        positiveCount = np.sum( np.count_nonzero( signMatrix == 1,axis=1), axis=1 )
        N = signMatrix.size/signMatrix.shape[0]
        return (positiveCount/N - 0.5)*np.sqrt(N)/0.5



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
    def errorTypeI( statistic, alpha, statisticType):
        sigma = np.std( statistic )
        zAlpha = scipy.stats.norm.ppf( Statistics.TEST_TYPE[statisticType](alpha) )
        thresholdForRejection = sigma * zAlpha
        occurences = np.where( np.abs(statistic)>thresholdForRejection )[0]

        return len(occurences)/len(statistic)

    @staticmethod
    def errorTypeII( statistic, alpha, statisticType):
        sigma = np.std( statistic )
        zAlpha = scipy.stats.norm.ppf( Statistics.TEST_TYPE[statisticType](alpha) )
        thresholdForRejection = sigma * zAlpha
        occurences = np.where( np.abs(statistic)<thresholdForRejection )[0]

        return len(occurences)/len(statistic)
