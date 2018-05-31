import numpy as np


def getStartAndEndOfWindow(window):
    whereIsWindowValid = np.where( window!=0 )[0]
    start = whereIsWindowValid[0]
    end = whereIsWindowValid[-1]
    return (start,end)

def generateSimpleWindow(start, end, total_length):
    if end>total_length or start>end:
        raise Exception("invalid indexes")
    
    window = np.zeros(total_length,dtype=int)
    window[start:end+1] = 1

    return window