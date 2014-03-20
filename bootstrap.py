import numpy as np
try:
    from scipy.stats import scoreatpercentile
except:
    # in case no scipy
    scoreatpercentile = False

def _confidence_interval_1d(A, confidenceLevel=.05, metric=np.mean, numResamples=10000, interpolate=True):
    """Calculates bootstrap confidence interval along one dimensional array"""
    N = len(A)
    resampleInds = np.random.randint(0, N, (numResamples,N))
    metricOfResampled = metric(A[resampleInds], axis=-1)
    if interpolate:
        lower = scoreatpercentile(metricOfResampled, confidenceLevel*100/2.0)
        upper = scoreatpercentile(metricOfResampled, 100-confidenceLevel*100/2.0)
    else:
        sortedMetricOfResampled = np.sort(metricOfResampled)
        lower = sortedMetricOfResampled[int(round(confidenceLevel*numResamples/2.0))]
        upper = sortedMetricOfResampled[int(round(numResamples - confidenceLevel*numResamples/2.0))]
    return lower, upper
    
def _ma_confidence_interval_1d(A, confidenceLevel=.05, metric=np.mean, numResamples=10000, interpolate=True):
    A = np.ma.masked_invalid(A, copy=True)
    A = A.compressed()
    lower, upper = _confidence_interval_1d(A, confidenceLevel, metric, numResamples, interpolate)
    return lower, upper

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
