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

def createSubDf(i, distDf, pearsonDf): # instead of concatinating why not create a new df?
    '''Create a sheet for a xlsx workbook'''
    pageName = "._.".join(map(str, i))
    combDf = pd.DataFrame({
        'Pearson_R': pearsonDf[i],
        'Corr_Dist': distDf[i]
    })
    return combDf, pageName

# def createXLSheet(distanceReport: pd.DataFrame, pearsonReport: pd.DataFrame, outName: str):
#     assert list(distanceReport.columns) == list(pearsonReport.columns)

#     print("Creating Excel file...", file=sys.stderr)
#     with pd.ExcelWriter(outName) as f:
#         for i in distanceReport.columns: #this actually takes so long to process, it might be better to cache some files
#             combDf, pageName = createSubDf(i, distDf=distanceReport, pearsonDf=pearsonReport)
#             combDf.to_excel(f, sheet_name=pageName)

#     print("...Completed!", file=sys.stderr)
#     return

import multiprocessing as mp
from openpyxl import load_workbook, Workbook
class CreateXLSheetMultithread:
    def __init__(self, cwd, distanceReport: pd.DataFrame, pearsonReport: pd.DataFrame, outName: str, threads: int = None):
        self.cwd = cwd
        self.distDf = distanceReport
        self.pearsonDf = pearsonReport
        self.threads = threads
        self.outName = outName

        self.folderPath = os.path.join(self.cwd, '.temp/')
    
    def __cleanTempFiles(self):
        folderPath = self.folderPath

        if os.path.exists(folderPath):
            for file in os.listdir(folderPath):
                filePath = os.path.join(folderPath, file)

                if os.path.isfile(filePath):
                    os.remove(filePath)
            os.rmdir(folderPath)
        return
    
    def __chunkToExcel(self, chunk: dict, outName: str):
        with pd.ExcelWriter(outName, engine='openpyxl') as writer:
            for sheetName, df in chunk.items():
                print(f'...Working on: {sheetName}...', file=sys.stderr)
                df.to_excel(writer, sheet_name=sheetName, index=True)
    
    def __processhandler(self, startIndex, endIndex, data: list, notebookDir):
        chunk = {d[1]:d[0] for d in data[startIndex:endIndex]}
        nbName = f'temp_{startIndex}.xlsx'
        name = os.path.join(self.cwd, '.temp/', nbName)

        self.__chunkToExcel(chunk=chunk, outName=name)
    
    def __createSubDf(self, i, distDf, pearsonDf):
        pageName = "._.".join(map(str, i))
        combDf = pd.DataFrame({
            'Pearson_R': pearsonDf[i],
            'Corr_Dist': distDf[i]
        })

        return combDf, pageName

    def parallelWrite(self):
        distanceReport = self.distDf
        pearsonReport = self.pearsonDf
        cwd = self.cwd

        try:
            os.mkdir('.temp/')
        except:
            self.__cleanTempFiles()
            os.mkdir('.temp/')
        
        sheets = [self.__createSubDf(i, distDf=distanceReport, pearsonDf=pearsonReport) for i in distanceReport.columns]
        numCPUs = (mp.cpu_count() - 2) if self.threads is None else (self.threads - 1) 
        chunkSize = len(sheets) // numCPUs

        print(f'Number of thread used in the program: {numCPUs} with 1 or 2 threads left as a spare', file=sys.stderr)
        print(f'Partition Size: {chunkSize}', file=sys.stderr)

        print('Beginning Multithreading process...', file=sys.stderr)

        processes = []
        for start in range(0, len(sheets), chunkSize):
            end = start + chunkSize if start + chunkSize <= len(sheets) else len(sheets)

            process = mp.Process(target=self.__processhandler, args=(start, end, sheets, self.cwd))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()

        print("...Multithreading process finished!", file=sys.stderr)
        
    def combNotebooks(self):
        tempNotebooks = []
        if os.path.exists(path=self.folderPath): # sanity check
            tempNotebooks = [os.path.join(self.folderPath, file) for file in os.listdir(self.folderPath)]
        else:
            print("\n.temp/ folder not found", file=sys.stderr)
            exit(-1)
        
        mergedWb = Workbook()
        mergedWb.remove(mergedWb.active)

        print("Merging all partitioned notebooks", file=sys.stderr)
        for notebook in tempNotebooks:
            workbook = load_workbook(filename=notebook, read_only=True)

            for sheet in workbook.sheetnames:
                ws = workbook[sheet]
                newWs = mergedWb.create_sheet(title = sheet)

                for row in ws.iter_rows():
                    newRow = [cell.value for cell in row]
                    newWs.append(newRow)

        mergedWb.save(self.outName)

        print("Merging done!", file=sys.stderr)
        self.__cleanTempFiles()


def main():
    from time import sleep
    from time import time
    allDF = pd.read_csv("data/AllReferenceExp_20191018_commonFeaturesOrder.csv",index_col=[0,1,2,3])
    testDF = pd.read_csv("data/SP0142_20171024_HeLa_10x_0_CP_histdiff_Concatenated.csv",index_col=[0,1,2,3])
    testDF = testDF.reindex(columns = allDF.columns)
    # print(allDF.to_numpy().shape)
    # cols= 1) (optional) sheet name 2) relationship to ref 3) dist 4) R
    testSig_comp = testDF.index[0]
    testSig = testDF.loc[testSig_comp]
    #print(testSig.to_numpy().shape[0])
    # print(len(allDF.iloc[0].to_numpy()))

    ## Above are test files ##
    ###### Main program below this commented line ######
    start = time()

    distance = pairwiseCorrProcess(testDF, allDF, distance=True)
    distance = distance.transpose()

    pearson = pairwiseCorrProcess(testDF, allDF, distance=False)
    pearson = pearson.transpose()

    xlsxgen = CreateXLSheetMultithread(cwd=os.getcwd(), distanceReport=distance, pearsonReport=pearson, outName='huh.xlsx', threads=8)
    xlsxgen.parallelWrite()
    sleep(1) # wait for files to flush to disk
    xlsxgen.combNotebooks()

    print(f'done! Completed in {time()-start}', file=sys.stderr)

if __name__ =='__main__':
    main()