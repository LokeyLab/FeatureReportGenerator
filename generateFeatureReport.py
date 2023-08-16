#!/usr/bin/env python3.5
########################################################################
# File:stainMerge.py
#  executable: generateFeatureReport.py
# Purpose:      Compute similarities and plate by plate stats on relatedness
#
# Author:       Akshar Lohith
# History:      AL 09/24/2019 Created
#
#   Need stainMerge
#
#
########################################################################
import sys, os, csv, subprocess
from datetime import date
from io import StringIO
import pandas as pd
from stainMerge import LastRun
from stainMerge import lastRunPickleObj

class CommandLine(object) :
    '''
    Handle the command line, usage and help requests.

    CommandLine uses argparse, now standard in 2.7 and beyond.
    it implements a standard command line argument parser with various argument options,
    a standard usage and help, and an error termination mechanism do-usage_and_die.

    attributes:
    myCommandLine.args is a dictionary which includes each of the available command line arguments as
    myCommandLine.args['option']

    methods:
    do_usage_and_die()
    prints usage and help and terminates with an error.
    '''

    def __init__(self, inOpts=None) :
        '''
        CommandLine constructor.
        Implements a parser to interpret the command line argv string using argparse.
        '''
        import argparse
        self.parser = argparse.ArgumentParser(\
            description = 'This program will perform stain set merging of histdiff '+\
            'files, using the CytoRunContentsFile as a guide for merge operations.',\
            add_help = True, #default is True
            prefix_chars = '-', usage = '%(prog)s')
        self.subparsers = self.parser.add_subparsers(dest='command',\
            help='Possible command group options.')
        self.subparsers.required = True

        # create subparser for performing operations on directory tree
        # meant for automatic processing.
        self.autoProcess = self.subparsers.add_parser("autoProcess",\
            aliases=['auto'],\
            help='Process stainSet merging following automatically used runcontents file.')

        self.autoProcess.add_argument("--runConFile", action = 'store',\
            default='runcontents.csv',\
            type=argparse.FileType('r', encoding='unicode_escape'),\
            help='RunContentsFile to automatically process with default parameters.\n'+\
            "If none given, the runcontentsfile.csv file in program's location "+\
            "will be used for  processing.")
        self.autoProcess.add_argument("--headersSearch", action = 'store',\
            default='reportCPfeatures.lst',\
            type=argparse.FileType('r', encoding='unicode_escape'),\
            help='Text file with the list of feature headers to excise out.')


        # create subparser for just updating the stored paths for the
        # stainMerge, current and future, operations.
        self.manualProcess = self.subparsers.add_parser("manualProcess",\
            aliases=['manual'],\
            help='Update the stored paths for stainMerge operations.')

        self.manualProcess.add_argument("MXzipPath", action = 'store',\
            type=os.path.abspath,\
            help='Path to dir of MX*.zip files.')
        self.manualProcess.add_argument("--ignoreMX", action = 'store',\
            default='reportFeatures_MXIgnore.lst',\
            type=argparse.FileType('r', encoding='unicode_escape'),\
            help='Text file with the list of MXids to ignore for report features.')
        self.manualProcess.add_argument('--runConFile',action='store',required=True,\
            default='runcontents.csv',\
            type=argparse.FileType('r', encoding='unicode_escape'),\
            help='RunContentsFile to manually seed the stored lastRun, or run alone.')
        self.manualProcess.add_argument("--force","-f", action='store_true',\
            help="Boolean flag to forcebly peform CP feature report of all " +\
            'experiments in provided RunContentsFile.')
        self.manualProcess.add_argument('--headersSearch',action='store',\
            default='reportCPfeatures.lst',\
            type=argparse.FileType('r', encoding='unicode_escape'),\
            help='Text file with the list of feature headers to excise out.')


        if inOpts is None :
            self.args = vars(self.parser.parse_args()) # parse the CommandLine options
        else:
            self.args = vars(self.parser.parse_args(inOpts)) # parse the input options

        self.args['headersSearch'] = [x.strip() for x in\
            self.args['headersSearch'].readlines()]

        if "auto" in self.args['command']:
            self.args['storedRun'] = lastRunPickleObj()
            verifiedNew = self.args['storedRun'].verifyLastRunConFile(\
                open('lastRunContents.txt','r').read().strip())
            if verifiedNew:
                self.args['storedRun'].updateLastRunCon(\
                    open('lastRunContents.txt','r').read().strip())
            else:
                sys.exit(0)
            # if

            self.args['ignoreMX'] = [x.strip() for x in\
                open('reportFeatures_MXIgnore.lst','r').readlines()]

            self.args['force'] = False

        elif "manual" in self.args['command']:
            self.args['storedRun'] = lastRunPickleObj()
            self.args['storedRun'].updatePath("Zip",self.args['MXzipPath'])
            if self.args['force'] and (self.args['runConFile'] is None):
                self.parser.error("--force requires --runConFile.")

            self.args['ignoreMX'] = [x.strip() for x in\
                self.args['ignoreMX'].readlines()]

            self.args['storedRun'].updateLastRunCon(self.args['runConFile'].name)

    def __del__ (self) :
        '''
        CommandLine destructor.
        '''
        # do something if needed to clean up before leaving
        pass

    def do_usage_and_die (self, str) :
        '''
        If a critical error is encountered, where it is suspected that the program is not being called with consistent parameters or data, this
        method will write out an error string (str), then terminate execution of the program.
        '''
        import sys
        print(str, file=sys.stderr)
        self.parser.print_usage()
        return 2


class Usage(Exception):
    '''
    Used to signal a Usage error, evoking a usage statement and eventual exit when raised.
    '''
    def __init__(self, msg):
        self.msg = msg

class Parser(object):
    """docstring for Parser."""
    def __init__(self, zipPath,outPath, runConFile=None):
        self.zipPath = zipPath
        self.outPath = outPath

    def cleanCNames(self,identifier):
        '''Function that removes the characters: _,.,(, ,),:, and -
            from a string, usually a header'''
        return identifier.replace(' ','').replace('(','').replace(')','').\
            replace('-','').replace(':','').replace('.','').replace("%","Pct")

    def correctColName(self, cName, EdUcheck, HD=False):
        ''' Append stainSet name to feature names as to know which stainset the
            feature refers to.'''
        if HD:
            if cName != 'Features':
                if EdUcheck ==1:
                    return cName+' (EdU)'
                elif EdUcheck == 0:
                    return cName+' (Cyto)'
            else:
                return cName
        else:
            if cName != 'Well Name':
                if EdUcheck ==1:
                    return cName+' (EdU)'
                elif EdUcheck == 0:
                    return cName+' (Cyto)'
            else:
                return cName

    def _wavelengthTransform(self,plateDetails,pmap,expDate):
        ''' collect the wavelength transformation names
            from the RunContentsFile'''
        runcon = self.runConFile
        #find the specific ex
        runcon_exp = runcon[\
            runcon['Measurement Name'].eq(plateDetails[0]) &\
            runcon['EdU (W2)'].eq(plateDetails[1]) &\
            runcon['Plate Map File'].eq(pmap) &\
            runcon['Experiment Date'].eq(expDate) &\
            runcon['Cell Lines'].eq(plateDetails[3]) &\
            runcon['Magnification'].apply(lambda s: s.lower()).eq(plateDetails[4]) &\
            runcon['TimePoint'].eq(plateDetails[5])]

        Wtransform = {
        'W1':runcon_exp['IXMW1'],
        'W2':runcon_exp['IXMW2'],
        'W3':runcon_exp['IXMW3'],
        'W4':runcon_exp['IXMW4']}

        # self.Wtransform = {'CYTO':None,'EDU':None}
        # # cyto = DAPI (1), ACTIN (3), TUBULIN (2), CALNEXIN (4)
        # # edu = DAPI (1), PH3 (3), EDU (2), GM130 (4)
        # self.Wtransform['CYTO'] = {'W1':'DAPI', 'W3':'ACTIN',
        #     'W2':'TUBULIN', 'W4':'CALNEXIN'}
        # self.Wtransform['EDU'] = {'W1':'DAPI', 'W3':'PH3',
        #     'W2':'EDU', 'W4':'GM130'}

        return Wtransform

    def performTransform(self, plateDetails,pmap,expDate,featureHeader):
        ''' convert feature header into human readable wavelength information'''
        Wtransform = self._wavelengthTransform(plateDetails,pmap,expDate)
        for key, value in Wtransform.items():
            if key in featureHeader:
                featureHeader.replace(key,value)
                return featureHeader


''' code blocks needed:
(SPplate,expDate,cytoPlate,eduPlate) = lastRun.lastRun
cytoPlate/eduPlate = (meas,cytoVedu, i, cellLine, mag,timePt)
os.path.join(\
self.lastRun.saveCPpath,\
'{}_{}_{}_{}_{}_CP_histdiff_Concatenated.csv'.\
format(pmap,exp.strftime('%Y%m%d'),cell,mag,timePt)))
{feat:cleanCNames(performTransform(cytoVedu,feat) for feat in headOut}
for ix in classCompositeSigs.index:
    print(ix)
    allDF.corrwith(classCompositeSigs.loc[ix],axis=1)
scipy.stats.mstats.pearsonr
wellKeys = set([chr(x)+str(z).zfill(2) for x in range(65,81) for z in range(1,25)])
scipy.spatial.distance.cdist #(https://stackoverflow.com/questions/47782104/compute-euclidean-distance-between-rows-of-two-pandas-dataframes?rq=1, https://docs.scipy.org/doc/scipy-0.17.1/reference/generated/scipy.spatial.distance.cdist.html)
'''
class DirParse(Parser):
    """docstring for DirParse."""
    def __init__(self,lastRun, zipPath, outPath, runConFile):
        self.lastRun = lastRun
        super(DirParse, self).__init__(path)
        walker = os.walk(self.zipPath)
        self.mxZips = next(walker)[-1]
        walker = os.walk(self.outPath)
        self.fingerprintPath = next(walker)[-1]

    def zipsProcess(self,files,path,headersSearch):
        '''Given a pair of MXids, find the MX*.zip files from the zipPath dir,
        read in and parse the header, (maybe convert to Human readable feat names)


        '''
        returnFiles = list()
        for filename, stain in files:
            #nameing of the new filtered featureList zip archive.
            newfileName = "FeatCount_MX"+ filename.split('MX')[-1]
            intermediateFile = os.path.join(path,newfileName.replace('.zip','.txt'))
            ''' Process a .zip file and aggregate well information based on
            self.method attribute. This function will save the given filename
            to a csv after aggregation or return to call the aggregated
            information as pandas DataFrame with 'save=False' option.'''

            print("processing",filename, file=sys.stderr)
            headOut = subprocess.run('unzip -p {} |head'.format(os.path.join(path,files[0])),\
                shell=True,stdout=subprocess.PIPE).\
                stdout.decode('raw_unicode_escape').split("\n")[0].strip().split('\t')
            indexKeep = [i+1 for i,h in enumerate(headOut) if h.replace('"','') in headersSearch]
            #format parsing command
            indexKeep[0] = "cut -f{}".format(indexKeep[0])
            indexKeep = [str(x) for x in indexKeep]
            cutCmd = ",".join(indexKeep)
            #parse the zip file for the features desired to be kept, removing intermediateFile after
            subprocess.run("unzip -p '{}' |{}>{} ;gzip {}".\
                format(os.path.join(path,filename),cutCmd,intermediateFile,intermediateFile),shell=True)

            #Example unix cmd to process a zip archive
            # """unzip -p cyto/MX4014.zip |cut -f1,2,3,4,5,7 >txzt.txt;zip txzt txzt.txt"""

            print("Processing Median Conversion",file=sys.stderr)
            try:
                df = pd.read_table(intermediateFile+'.gz')#,compression="zip")
            except:
                print("Error handling zip file. will try to process unzipped file.",\
                        file=sys.stderr)
                # using StringIO, pipe the subprocess stdout stream into
                # pandas read_table function
                df = pd.read_table(StringIO(subprocess.run('gunzip -c '+\
                    intermediateFile+'.gz', shell=True,\
                    stdout=subprocess.PIPE).stdout.decode('raw_unicode_escape')))
            #aggregate the well information by median and save as csv.
            dagg = df.groupby("Well Name").agg('median')
            # dagg.to_csv(os.path.join(path,newfileName.replace('.zip',"_WELL.csv")))
            returnFiles.append(dagg)
            subprocess.run(['rm', intermediateFile+'.gz'],shell=False)
        return returnFiles

    def medianFeatureExtract(self,filename,headersSearch,path,plateDetails,pmap,expDate):
        '''perform median processing and select feature extraction on the given MX file
        '''

        #nameing of the new filtered featureList zip archive.
        newfileName = "FeatCount_MX"+ filename.split('MX')[-1]
        intermediateFile = os.path.join(path,newfileName.replace('.zip','.txt'))

        #determine the header indexes to be collected.
        print("processing",filename, file=sys.stderr)
        headOut = subprocess.run('unzip -p {} |head'.format(os.path.join(path,filename)),\
            shell=True,stdout=subprocess.PIPE).\
            stdout.decode('raw_unicode_escape').split("\n")[0].strip().split('\t')

        headOut_desired = [self.performTransform(plateDetails,pmap,expDate,x) for x in headOut]

        indexKeep = [i+1 for i,h in enumerate(headOut_desired) if h.replace('"','') in headersSearch]

        #format parsing command
        indexKeep[0] = "cut -f{}".format(indexKeep[0])
        indexKeep = [str(x) for x in indexKeep]
        cutCmd = ",".join(indexKeep)
        #parse the zip file for the features desired to be kept, removing intermediateFile after
        subprocess.run("unzip -p '{}' |{}>{} ;gzip {}".\
            format(os.path.join(path,filename),cutCmd,intermediateFile,intermediateFile),shell=True)

        #Example unix cmd to process a zip archive
        # """unzip -p cyto/MX4014.zip |cut -f1,2,3,4,5,7 >txzt.txt;zip txzt txzt.txt"""

        print("Processing Median Conversion",file=sys.stderr)
        try:
            df = pd.read_table(intermediateFile+'.gz')#,compression="zip")
        except:
            print("Error handling zip file. will try to process unzipped file.",\
                    file=sys.stderr)
            # using StringIO, pipe the subprocess stdout stream into
            # pandas read_table function
            df = pd.read_table(StringIO(subprocess.run('gunzip -c '+\
                intermediateFile+'.gz', shell=True,\
                stdout=subprocess.PIPE).stdout.decode('raw_unicode_escape')))

        #aggregate the well information by median and save as csv.
        dagg = df.groupby("Well Name").agg('median')
        # dagg.to_csv(os.path.join(path,newfileName.replace('.zip',"_WELL.csv")))
        subprocess.run(['rm', intermediateFile+'.gz'],shell=False)
        return dagg

    def fingerprintProcess(self):
        '''Go through the HistDiff_Concatenated_CP directory, collect the reference
        experiments, concatenate them and hold for comparison. As each experiment
        is processed from the runConFile, pull the already processed histdiff_cp CSV
        and perform pairwise PearsonCorr calcs between each comp in the processed exp
        and the reference experiments. Return to call the DF with all compound
        identifier columns + 2 columns with 1) similarity scores as a vector and
        2) the vector of compound identifiers
        '''
        pass

    def pairwiseCorrProcess(exp_df,ref_df,reporting_df=None):
        '''function to take given experiment df and the reference DF to perform pairwise
        correlation simiarity and distance calculations for each row of the experiment df
        to each compound in the reference DF. A reporting df for the experiement is either
        created, if one is not passed, or appended with 3 columns:
        1) reference Compound identifiers
        2) corresponding Pearson similarity Correlations (as 10 digit floats)
        3) corresponding correlation distances (as 10 digit floats)
        Return to call the reporting DF.
        https://stackoverflow.com/a/54655688 source for solution
        NOTE: order of the features in each DF matter for the distance and similarity calcs
        '''

        from scipy.spatial import distance as sp_distance
        from scipy import stats as sp_stats

        #define local functions that will compute the pairwise correlation distance
        def corrDist (compSig:np.ndarray, refArray:np.ndarray) -> np.ndarray:
            returnDists = [0]*refArray.shape[0]
            for i in range(refArray.shape[0]):
                x = refArray[i,:]
                returnDists[i] = sp_distance.correlation(compSig, x)
            return np.array(returnDists)

        #define local functions that will compute the pairwise correlation similarity
        def pearsonr (compSig:np.ndarray, refArray:np.ndarray) -> np.ndarray:
            returnCorrs = [0]*refArray.shape[0]
            for i in range(refArray.shape[0]):
                x = refArray[i,:]
                returnCorrs[i] = sp_stats.pearsonr(x, compSig)[0]
            return np.array(returnCorrs)

        if reporting_df is None:
            reporting_df = pd.DataFrame(columns=["TopSimilarRefComps","CorrespPearsonSim","CorrespCorrDistance"], index=exp_df.index)

        # one_to_one = list() # for sanity checks
        refcompounds = ["._.".join([str(y) for y in x]) for x in ref_df.index]
        # print(ref_df.to_numpy().shape)

        distance = exp_df.apply(lambda compSig: corrDist(compSig.to_numpy(),ref_df.to_numpy()),axis=1)
        similarity = exp_df.apply(lambda compSig: pearsonr(compSig.to_numpy(),ref_df.to_numpy()),axis=1)

        #format the reference compound similarities in the report dataframe and return to call
        reporting_df['TopSimilarRefComps'] = ".__.".join(refcompounds)
        reporting_df['CorrespCorrDistance'] = \
                [".__.".join(["{:.10f}".format(d_1) for d_1 in d]) for d in distance]
        reporting_df['CorrespPearsonSim'] = \
                [".__.".join(["{:.10f}".format(s_1) for s_1 in s]) for s in similarity]

        return reporting_df #, one_to_one #for sanity checks



def main(myCommandLine =None):
    if myCommandLine is None:
        myCommandLine = CommandLine()
    else :
        myCommandLine = CommandLine(['-h'])
    parseCmd =myCommandLine.args
    # print(parseCmd)
    # try:
    #     if parseCmd['plateMap']:
    #         print("plateMap is: {}".format(parseCmd['plateMap']), file=sys.stderr)
    # except KeyError:
    #     pass
    operation = DirParse(path=parseCmd['storedRun'].MXFileLoc, \
        lastRun=parseCmd['storedRun'], runConFile=parseCmd['runConFile'],\
        keepColumns=parseCmd['keepFeatures'])
    operation.concatPlates(force=parseCmd['force'])

if __name__ == "__main__":
    # if program is launched alone, this is true and is exececuted. if not, nothing is\
    # executedf rom this program and instead objects and variables are made available \
    # to the program that imports this.
    main();
    raise SystemExit
