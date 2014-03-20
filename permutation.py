import numpy as np

def _p_val_1d(A, B, metric=np.mean, numResamples=10000):
    """Return p value of observed difference between 1-dimensional A and B"""
    observedDiff = abs(metric(A) - metric(B))
    combined = np.concatenate([A, B])
    numA = len(A)
    
    resampleDiffs = np.zeros(numResamples,dtype='float')
    for resampleInd in range(numResamples):
        permutedCombined = np.random.permutation(combined)
        diff = metric(permutedCombined[:numA]) - metric(permutedCombined[numA:])
        resampleDiffs[resampleInd] = diff
    
    pVal = (np.sum(resampleDiffs > observedDiff) + np.sum(resampleDiffs < -observedDiff))/float(numResamples)
    
    return pVal
    
def _ma_p_val_1d(A, B, metric=np.mean, numResamples=10000):
    A = np.ma.masked_invalid(A, copy=True)
    A = A.compressed()
    B = np.ma.masked_invalid(B, copy=True)
    B = B.compressed()
    pVal = _p_val_1d(A, B, metric, numResamples)
    return pVal

def _ma_p_val_concatenated(C, numSamplesFirstGroup, metric=np.mean, numResamples=10000):
    A = C[:numSamplesFirstGroup]
    B = C[numSamplesFirstGroup:]
    pVal = _ma_p_val_1d(A, B, metric, numResamples)
    return pVal

def p_val(A, B, axis=None, metric=np.mean, numResamples=10000):
    """Return the p value that metric(A) and metric(B) differ along an axis ignoring NaNs and masked elements.
    
    Parameters
    ----------
    A : array_like
        Array containing numbers of first group. 
    B : array_like
        Array containing numbers of second group. 
    axis : int, optional
        Axis along which the p value is computed.
        The default is to compute the p value of the flattened arrays.
    metric : numpy function, optional
        metric to calculate p value for.
        The default is numpy.mean
    numResamples : int, optional
        number of permutations. The default is 10000.
        
    Returns
    -------
    pValue : ndarray
    An array with the same shape as `A` and `B`, with the specified axis removed.
    If axis is None, a scalar is returned.
    """
        
    A = A.copy()
    B = B.copy()
    if axis is None:
        A = A.ravel()
        B = B.ravel()
        pVal = _ma_p_val_1d(A, B, metric, numResamples)
    else:
        numSamplesFirstGroup = A.shape[axis]    
        C = np.concatenate((A,B),axis=axis)
        pVal = np.apply_along_axis(_ma_p_val_concatenated, axis, C, numSamplesFirstGroup, metric, numResamples)
        
    return pVal
