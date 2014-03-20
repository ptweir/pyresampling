import numpy as np
import collections
try:
    from scipy.stats import scoreatpercentile
except:
    # in case no scipy
    scoreatpercentile = False

def _confidence_interval_1d(A, confidenceLevel=.05, metric=np.mean, numResamples=10000, interpolate=True):
    """Calculates bootstrap confidence interval along one dimensional array"""
    
    if not isinstance(confidenceLevel, collections.Iterable):
        confidenceLevel = np.array([confidenceLevel])

    N = len(A)
    resampleInds = np.random.randint(0, N, (numResamples,N))
    metricOfResampled = metric(A[resampleInds], axis=-1)

    confidenceInterval = np.zeros(2*len(confidenceLevel),dtype='float')
    
    if interpolate:
        for thisConfidenceLevelInd, thisConfidenceLevel in enumerate(confidenceLevel):
            confidenceInterval[2*thisConfidenceLevelInd] = scoreatpercentile(metricOfResampled, thisConfidenceLevel*100/2.0)
            confidenceInterval[2*thisConfidenceLevelInd+1] = scoreatpercentile(metricOfResampled, 100-thisConfidenceLevel*100/2.0)
    else:
        sortedMetricOfResampled = np.sort(metricOfResampled)
        for thisConfidenceLevelInd, thisConfidenceLevel in enumerate(confidenceLevel):
            confidenceInterval[2*thisConfidenceLevelInd] = sortedMetricOfResampled[int(round(thisConfidenceLevel*numResamples/2.0))]
            confidenceInterval[2*thisConfidenceLevelInd+1] = sortedMetricOfResampled[int(round(numResamples - thisConfidenceLevel*numResamples/2.0))]
    return confidenceInterval
    
def _ma_confidence_interval_1d(A, confidenceLevel=.05, metric=np.mean, numResamples=10000, interpolate=True):
    A = np.ma.masked_invalid(A, copy=True)
    A = A.compressed()
    confidenceInterval = _confidence_interval_1d(A, confidenceLevel, metric, numResamples, interpolate)
    return confidenceInterval

def confidence_interval(A, axis=None, confidenceLevel=.05, metric=np.mean, numResamples=10000, interpolate=True):
    """Calculates bootstrap confidence interval along the given axis"""
    
    if interpolate is True and scoreatpercentile is False:
        print "need scipy to interpolate between values"
        interpolate = False
        
    A = A.copy()
    if axis is None:
        A = A.ravel()
        outA = _ma_confidence_interval_1d(A, confidenceLevel, metric, numResamples, interpolate)
    else:
        outA = np.apply_along_axis(_ma_confidence_interval_1d, axis, A, confidenceLevel, metric, numResamples, interpolate)
        
    return outA
