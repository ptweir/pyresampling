import numpy as np
import math
import itertools

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
    
    See Also
    --------
    fast_p_val : uses the same random permutation for all entries.
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
    
def fast_p_val(A, B, axis=0, metric=np.mean, numResamples=10000):
    """Return the p value that metric(A) and metric(B) differ along an axis.
    
    Parameters
    ----------
    A : array_like
        Array containing numbers of first group. 
    B : array_like
        Array containing numbers of second group. 
    axis : int, optional
        Axis along which the p value is computed.
        The default is to compute the p value along the first dimension.
    metric : numpy function, optional
        metric to calculate p value for.
        The default is numpy.mean
    numResamples : int, optional
        number of permutations. The default is 10000.
        
    Returns
    -------
    pValue : ndarray
        An array with the same shape as `A` and `B`, with the specified axis removed.
    
    See Also
    --------
    p_val : ignores NaNs and masked elements, and independently calculates random
            permutations for each entry.
    """
    rolledA = np.rollaxis(A,axis)
    rolledB = np.rollaxis(B,axis)
    
    observedDiff = abs(metric(rolledA,axis=0) - metric(rolledB,axis=0))
    combined = np.concatenate((rolledA, rolledB),axis=0)
    numA = np.shape(rolledA)[0]
    
    resampleShape = np.array(combined.shape)
    resampleShape[0] = numResamples
    resampleDiffs = np.zeros(resampleShape,dtype='float')
    for resampleInd in range(numResamples):
        permutedCombined = np.random.permutation(combined)
        diff = metric(permutedCombined[:numA,...],axis=0) - metric(permutedCombined[numA:,...],axis=0)
        resampleDiffs[resampleInd,...] = diff
        
    pVal = (np.sum(resampleDiffs > observedDiff,axis=0) + np.sum(resampleDiffs < -observedDiff,axis=0))/float(numResamples)
        
    return pVal

def exact_p_val(A, B, axis=0, metric=np.mean):
    """Return the p value that metric(A) and metric(B) differ along an axis. By computing all
    permutations of the union of A and B with corresponding sizes
    
    Parameters
    ----------
    A : array_like
        Array containing numbers of first group. 
    B : array_like
        Array containing numbers of second group. 
    axis : int, optional
        Axis along which the p value is computed.
        The default is to compute the p value along the first dimension.
    metric : numpy function, optional
        metric to calculate p value for.
        The default is numpy.mean
        
    Returns
    -------
    pValue : ndarray
        An array with the same shape as `A` and `B`, with the specified axis removed.
    
    See Also
    --------
    p_val : for larger sample sizes, in which computing all permutations is infeasible
    """
    rolledA = np.rollaxis(A,axis)
    rolledB = np.rollaxis(B,axis)
    
    observedDiff = abs(metric(rolledA,axis=0) - metric(rolledB,axis=0))
    combined = np.concatenate((rolledA, rolledB),axis=0)
    numA = np.shape(rolledA)[0]
    numB = np.shape(rolledB)[0]
    
    resampleShape = np.array(combined.shape)
    resampleShape[0] = math.factorial(numA + numB)/(math.factorial(numA)*math.factorial(numB))
    resampleDiffs = np.zeros(resampleShape,dtype='float')

    for resampleInd, indsToInclude in enumerate(itertools.combinations(range(numA + numB), numA)):
        indsToExclude = [i for i in range(12) if i not in indsToInclude]
        permutedA = combined[indsToInclude, ...]
        permutedB = combined[indsToExclude, ...]
        diff = metric(permutedA, axis=0) - metric(permutedB, axis=0)
        resampleDiffs[resampleInd,...] = diff
        
    pVal = (np.sum(resampleDiffs > observedDiff, axis=0) + np.sum(resampleDiffs < -observedDiff, axis=0))/float(resampleShape[0])
        
    return pVal
