import sys, os, subprocess
from datetime import date
from io import StringIO
import pandas as pd, numpy as np
import time, timeit, cython
from numba import jit, prange, njit
from scipy.spatial import distance as sp_distance
from scipy import stats as sp_stats

@jit(nogil=True, nopython=True)
def correlation(u, v, w=None, centered=True): #rip from scipy.spatial.distance source
    """
    Compute the correlation distance between two 1-D arrays.
    The correlation distance between `u` and `v`, is
    defined as
    .. math::
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}
    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0
    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.
    """
    if centered:
        umu = np.mean(u) 
        vmu = np.mean(v)
        u = u - umu
        v = v - vmu
    uv = np.mean(u * v) 
    uu = np.mean(np.square(u))
    vv = np.mean(np.square(v))
    dist = 1.0 - uv / np.sqrt(uu * vv)
    return dist

@njit(nogil=True, parallel = True)
def numbaParallelCorrDistonRef(compSig,refArray):
    """attempt at making a nice numba accelerated version of CorrDist function,
    by parallelizing the specific compDist calculation to refSigs."""

    # I need to refactor the code (it's too lengthy to read)
    numRefSigs = refArray.shape[0]
    returnDists = np.zeros(refArray.shape)

    compSigMinusMean = compSig - np.mean(compSig)
    compSigMeanSqr = np.mean(np.square(compSigMinusMean))

    refMinusMean = lambda i: refArray[i] - np.mean(refArray[i])
    refMeanSqr = lambda i: np.mean(np.square(refMinusMean(i)))

    for i in prange(numRefSigs):
        returnDists[i] = (1.0 - np.mean(compSigMinusMean * refMinusMean(i)) / np.sqrt(compSigMeanSqr * refMeanSqr(i)))

    
    return returnDists

def main():
    allDF = pd.read_csv("data/AllReferenceExp_20191018_commonFeaturesOrder.csv",index_col=[0,1,2,3])
    testDF = pd.read_csv("data/SP0142_20171024_HeLa_10x_0_CP_histdiff_Concatenated.csv",index_col=[0,1,2,3])
    testDF = testDF.reindex(columns = allDF.columns)
    print(allDF.to_numpy().shape)

    testSig_comp = testDF.index[0]
    testSig = testDF.loc[testSig_comp]
    #print(testSig.to_numpy().shape[0])
    print(len(allDF.iloc[0].to_numpy()))

    def test():
        return numbaParallelCorrDistonRef(testSig.values, allDF.to_numpy())[0]

    print(timeit.timeit(test, number=10000))


if __name__ =='__main__':
    main()