import sys, os, subprocess
from datetime import date
from io import StringIO
import pandas as pd, numpy as np
import time, timeit, cython
from numba import jit, prange, njit

@jit(nopython=True, parallel=True)
def corrDist(compSig, refArray):
    def cor(u,v, centered=True): # ripped from scipy.spatial.distances
        if centered:
            umu = np.average(u)
            vmu = np.average(v)
            u = u - umu
            v = v - vmu
        uv = np.average(u*v)
        uu = np.average(np.square(u))
        vv = np.average(np.square(v))
        dist = 1 - uv / np.sqrt(uu*vv)
        return np.abs(dist)

    num_ref_sigs = refArray.shape[0]
    return_dists = np.empty(num_ref_sigs)
    
    for i in prange(num_ref_sigs):
        x = refArray[i]
        return_dists[i] = cor(compSig, x)
    
    return return_dists

@jit(nopython=True, parallel=True)
def pearsonr(compSig, refArray):
    num_ref_sigs = refArray.shape[0]
    return_corrs = np.empty(num_ref_sigs)

    for i in prange(num_ref_sigs):
        x = refArray[i,:]
        return_corrs[i] = np.corrcoef(x, compSig)[0,1]
    return return_corrs


def pairwiseCorrProcess(exp_df, ref_df, reporting_df=None, distance = True):
    if reporting_df is None:
        reporting_df = pd.DataFrame(columns=ref_df.index, index=exp_df.index)
    
    if distance:
        data = exp_df.apply(lambda compSig:\
            corrDist(compSig=compSig.to_numpy(), refArray=ref_df.to_numpy()).tolist(),\
            axis=1)
        # cols are the refIndex name
        data = data.apply(pd.Series)
        data.columns = list(ref_df.index)
        reporting_df = pd.DataFrame(data, columns=ref_df.index)
    else: 
        data = exp_df.apply(lambda compSig:\
            pearsonr(compSig=compSig.to_numpy(), refArray=ref_df.to_numpy()),\
            axis=1)
        data = data.apply(pd.Series)
        data.columns = list(ref_df.index)
        reporting_df = pd.DataFrame(data, columns=ref_df.index)
            
    return reporting_df

def main():
    allDF = pd.read_csv("data/AllReferenceExp_20191018_commonFeaturesOrder.csv",index_col=[0,1,2,3])
    testDF = pd.read_csv("data/SP0142_20171024_HeLa_10x_0_CP_histdiff_Concatenated.csv",index_col=[0,1,2,3])
    testDF = testDF.reindex(columns = allDF.columns)
    print(allDF.to_numpy().shape)
    # cols= 1) (optional) sheet name 2) relationship to ref 3) dist 4) R
    testSig_comp = testDF.index[0]
    testSig = testDF.loc[testSig_comp]
    #print(testSig.to_numpy().shape[0])
    print(len(allDF.iloc[0].to_numpy()))

    final = pairwiseCorrProcess(testDF, allDF, distance=False)
    print(final.head(5))
    #final.to_csv('bruh2.csv', sep=',')
    # def test():
    #     return corrDist(testSig.values, allDF.to_numpy())
    #     # return numbaParallelCorrDistonRef(testSig.values, allDF.to_numpy())[0]

    # print(timeit.timeit(test, number=7))


if __name__ =='__main__':
    main()