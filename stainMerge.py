#!/usr/bin/env python3.5
########################################################################
# File:stainMerge.py
#  executable: stainMerge.py
# Purpose:      Merge Stain sets to form RAW CP fingerprint based on RunContentsFile
#
# Author:       Akshar Lohith
# History:      AL 05/31/2019 Created
#               AL 06/03/2019 Core concatenation function complete (still need
#                   commandLine handler)
#               AL 09/13/2019 common features set determined and enabled.
#                   Commandline handler complete, program operational.
#
########################################################################
import sys, os, csv
from datetime import date
import pandas as pd

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

        # create subparser for just updating the stored paths for the
        # stainMerge, current and future, operations.
        self.manualProcess = self.subparsers.add_parser("manualProcess",\
            aliases=['manual'],\
            help='Update the stored paths for stainMerge operations.')

        self.manualProcess.add_argument("histDiffMXpath", action = 'store',\
            type=os.path.abspath,\
            help='Path to dir of MX*.histdiff.csv files.')
        self.manualProcess.add_argument("plateMapsPath", action = 'store',\
            type=os.path.abspath,\
            help="Path to dir of CytoPlateMap csv's.")
        self.manualProcess.add_argument('--runConFile',action='store',required=True,\
            default='runcontents.csv',\
            type=argparse.FileType('r', encoding='unicode_escape'),\
            help='RunContentsFile to manually seed the stored lastRun, or run alone.')
        self.manualProcess.add_argument("--force","-f", action='store_true',\
            help="Boolean flag to forcebly peform stain merge of all experiments "+\
            'in provided RunContentsFile.')
        self.manualProcess.add_argument('--keepFeatures',action='store',\
            type=argparse.FileType('r',encoding='unicode_escape'),\
            help='Keep features list in the form of a .lst file.')
        self.manualProcess.add_argument('--recentRun',action='store_true',\
            help='Boolean flag to print out last merged file. Reads and displays pickle contents.')

        if inOpts is None :
            self.args = vars(self.parser.parse_args()) # parse the CommandLine options
        else:
            self.args = vars(self.parser.parse_args(inOpts)) # parse the input options

        if "auto" in self.args['command']:
            self.args['storedRun'] = lastRunPickleObj()
            verifiedNew = self.args['storedRun'].verifyLastRunConFile(\
                open('lastRunContents.txt','r').read().strip())
            if verifiedNew:
                self.args['storedRun'].updateLastRunCon(\
                    open('lastRunContents.txt','r').read().strip())
            else:
                sys.exit(0)
            self.args['force'] = False
            self.args['keepFeatures'] = [x.strip() for x in\
                open('commonFeatures.lst','r').readlines()]

        elif "manual" in self.args['command']:
            if self.args['recentRun']:
                self.args['storedRun'] = lastRunPickleObj()
                print("Last MX pair processed:")
                print(self.args['storedRun'].lastRun)
                print("Last runcontents file used: {}".\
                    format(self.args['storedRun'].lastRunConFile))
                sys.exit(0)
            elif self.args['force'] and (self.args['runConFile'] is None):
                self.parser.error("--force requires --runConFile.")
            self.args['keepFeatures'] = [x.strip() for x in\
                self.args['keepFeatures'].readlines()]
            # print(self.args['keepFeatures'],file=sys.stderr)
            self.args['storedRun'] = lastRunPickleObj(update=\
                (self.args['plateMapsPath'],self.args['histDiffMXpath']))
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

class LastRun(object):
    """
        Object containing filepaths and lastRun details that can be updated.
        These paths are essentially strings, and the current locations of specific
        file directories, such as CytoPlateMaps, or MX*.zip files etc.
        The Paths object will be un/pickled and checked first thing upon program
        execution such that all assumed or needed paths have been stored for program usage.
    """
    def __init__(self, PlateMapsLoc, MXFileLoc, IXMLoc=None):
        self.updatePath("PlateMap",PlateMapsLoc)
        self.updatePath("MX", MXFileLoc)
        #currently set to None, future option to enable to also store lastRun IXM location
        # self.IXMLoc = IXMLoc

    def updatePath(self,pathType,path):

        if not os.path.exists(os.path.abspath(path)):
            pathType = False

        if pathType == 'IXM':
            self.IXMLoc = path
        elif pathType == 'MX':
            self.MXFileLoc = path
            if not os.path.exists(\
                os.path.join(self.MXFileLoc,'HistDiff_Concatenated_RAW')):
                os.mkdir(os.path.join(self.MXFileLoc,'HistDiff_Concatenated_RAW'))

            self.saveRawPath = os.path.join(self.MXFileLoc,'HistDiff_Concatenated_RAW')

            if not os.path.exists(\
                os.path.join(self.MXFileLoc,'HistDiff_Concatenated_CP')):
                os.mkdir(os.path.join(self.MXFileLoc,'HistDiff_Concatenated_CP'))

            self.saveCPpath = os.path.join(self.MXFileLoc,'HistDiff_Concatenated_CP')

        elif pathType == 'PlateMap':
            self.PlateMapsLoc = path
        elif pathType == 'Zip':
            self.MXzipLoc = path
        else:
            print("Incorrect option provided, or invalid path to be stored.",\
                file=sys.stderr)

    def recordLastRun(self,cytoPlate,eduPlate,SPplate,expDate):
        ''' update the last SPplate_exp stains merged for the next time the
            program is run
        '''
        if not isinstance(expDate,date):
            expDate = date(year=int(expDate[0:4]),
                month=int(expDate[4:6]),
                day=int(expDate[6:]))
        self.lastRun = (SPplate,expDate,cytoPlate,eduPlate)

    def recordLastRunContentsDict(self,runConDict):
        '''store the MX file Pairings for a filter features program'''
        self.lastProcessedPairs = runConDict

    def updateLastRunCon(self,lastRunCon):
        print("storing lastRunConFile",file=sys.stderr)
        self.lastRunConFile = lastRunCon

    def verifyLastRunConFile(self,testRunCon):
        '''return a verification if the test RunContentsFile is different than
            the stored RunContentsFileself.
        '''
        return os.path.basename(self.lastRunConFile) != os.path.basename(testRunCon)

class Parser(object):
    """docstring for Parser."""
    def __init__(self, path):
        self.path = path

    def cleanCNames(self,identifier):
        '''Function that removes the characters: _,.,(, ,),:, and -
            from a string, usually a header'''
        return identifier.replace(' ','').replace('(','').replace(')','').\
            replace('-','').replace(':','').replace('.','')

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

class DirParse(Parser):
    """docstring for DirParse."""
    def __init__(self, path, lastRun, runConFile=None,keepColumns=None):
        self.runConFile = runConFile
        self.lastRun = lastRun
        super(DirParse, self).__init__(path)
        walker = os.walk(self.path)
        self.files = next(walker)[-1]
        self.keepColumns = keepColumns
        if isinstance(self.keepColumns,list):
            insertList = list()
            #'CompoundName' #CompoundName will be the index so don't need
            # in the columnslicer
            for i,idCol in enumerate(\
                ['Concentration','PlateName','WellID']):

                if idCol not in self.keepColumns:
                    insertList.append(idCol)

            insertList.extend(self.keepColumns)
            self.keepColumns = insertList

    def _wellFinder(self,df,pmap,emergencyFill):
        ''' find the corresponding platemap file to the df and add column with
            compound wellID. Sort the columns so compound identifiers are first
            followed by the RAW CP fingerprint
        '''
        from numpy import round
        #read in plateMapFile, and unpack into dictionary of compound keys,
        # and list of well+concentration tuples values
        pmap_name = pmap
        try:
            pmap = pd.read_csv(\
                os.path.join(self.lastRun.PlateMapsLoc,'{}.csv'.format(pmap)))
            stupidEncoding=False

        except UnicodeDecodeError:
            #Becaause some special characters have made there way into the pmap
            #need to read in with different decoding scheme
            #its assumed that latin-1 is correct, but a unix terminal command of
            #'file -i [filename]' will tell what kind of actual encoding..
            pmap = pd.read_csv(\
                os.path.join(self.lastRun.PlateMapsLoc,'{}.csv'.format(pmap)),\
                encoding='raw_unicode_escape')
            stupidEncoding=True
            print('Tripped Stupid Decode Error',file=sys.stderr)

        except FileNotFoundError:
            print("PlateMap missing from PlateMaps Directory",file=sys.stderr)
            return None
        #assuming if no listed concentration that conc is -1,
        # pd.DataFrame.fillna is just easy way to get there
        pmap.fillna(-1,inplace=True)
        pmap_mapping = dict()
        wellIdentifierColName =\
            ['384 well','384well','384 Well','Well Name', 'Well','384 well ']
        compoundHeader = [col for col in pmap.columns if 'Molecule' in col][0]
        concHeader = [col for col in pmap.columns if \
            'Concentration' in col or 'Molarity' in col][0]
        for chaos in wellIdentifierColName:
            if chaos in pmap.columns:
                for i,(compName,conc,wellLoc) in enumerate(\
                pmap[[compoundHeader,concHeader,chaos]].values):
                    # if stupidEncoding:
                    #     if compName not in pmap_mapping.keys():
                    #         pmap_mapping[str(compName).strip().decode('latin-1')]= [(float(conc),wellLoc.decode('latin-1'))]
                    #     else:
                    #         pmap_mapping[str(compName).strip().decode('latin-1')].append((float(conc),wellLoc.decode('latin-1')))
                    # else:
                    if compName not in pmap_mapping.keys():
                        pmap_mapping[str(compName)]= [(float(conc),wellLoc)]
                    else:
                        pmap_mapping[str(compName)].append((float(conc),wellLoc))
        # print(pmap_mapping,file=sys.stderr)
        convertedWellID = list()
        wellMapped = set()
        # print(pmap_mapping.keys(),"\n\n")
        print("index","Compound","searchConc","checkWellSeen","potentialWell","checkConcIsSearchConc","potentialConc",pmap_name,file=sys.stderr)
        for i,x in enumerate(df.index):
            searchConc = float(df.iloc[i].Concentration)
            if 'e-' in str(searchConc).split(".")[-1]:
                searchConcLen = None
            else:
                searchConcLen = len(str(searchConc).split(".")[-1])
            # print(i,x,searchConc,file=sys.stderr)
            try:
                potentialWells = pmap_mapping[x]

                # .loc[\
                # (pmap[compoundHeader]==x) & (pmap[concHeader] ==\
                #     df.loc[x,'Concentration']),chaos].values
                for conc,well in potentialWells:
                    if searchConcLen is not None:
                        if (well not in wellMapped) and (float(searchConc) == float(format(conc,".{}f".format(searchConcLen)))):
                            convertedWellID.append(well)
                            wellMapped.add(well)
                            pmap_mapping[x].remove((conc,well))
                            # print("compound {} mapped to {} in {}".format(x,well,pmap_name),file=sys.stderr)
                            print("compound {},search conc {} mapped to {}, conc {} in {}".format(x,searchConc,well,conc,pmap_name),file=sys.stderr)
                            break
                    else:
                       if (well not in wellMapped) and (float(format(float(searchConc),'e')) == float(format(float(conc),'e'))):
                           convertedWellID.append(well)
                           wellMapped.add(well)
                           pmap_mapping[x].remove((conc,well))
                           print("compound {},search conc {:e} mapped to {}, conc {:e} in {}".format(x,searchConc,well,conc,pmap_name),file=sys.stderr)
                           break
                    testWell = (well not in wellMapped)
                    testConc = (float(searchConc) == float(format(conc,".{}f".format(searchConcLen))))
                    print(i,x,searchConc,testWell,well,testConc,float(conc),file=sys.stderr)
                    # else:
                    #     print("Can't map {}_{} from {}".\
                    #         format(x,searchConc,pmap_name),file=sys.stderr)

                if i != (len(convertedWellID)-1):
                    print(i,x,pmap_name,potentialWells,searchConc,searchConcLen,end="\n\n",file=sys.stderr)
                    convertedWellID.append(emergencyFill)
            except KeyError:
                convertedWellID.append("NOTFOUND_COMPMAPERROR_{}".format(emergencyFill))
            except ValueError:
                convertedWellID.append("NOTFOUND_CONCERROR_{}".format(emergencyFill))
        if len(df.index) != len(convertedWellID):
            print('compoundIndex != convertedWell: {}'.format(pmap_name),\
            len(df.index),len(convertedWellID),"\n\n\n\n",file=sys.stderr)
        df['WellID'] = convertedWellID

        #reorder the columns so the compound identifier columns come first
        cols = df.columns.tolist()
        IDcols = ['Concentration','PlateName','WellID']
        cols = [c for c in cols if c not in IDcols]
        IDcols.extend(cols)
        return df[IDcols]

    def _parseRunCon(self):
        ''' parse runconfile and return mapping dictionary '''
        import collections

        runcon = pd.read_csv(self.runConFile)
        if len(runcon.loc[runcon['Experiment Date'] == 2011001].index) >0:
            for ix in runcon.loc[runcon['Experiment Date'] == 2011001].index:
                runcon.loc[ix,'Experiment Date'] = 20111001
        #hardCodeFix for SP20063 and SP20065
        # for fxmap in ['SP20063','SP20065']:
        #     checkDate = runcon.loc[runcon['Plate Map File']==fxmap,'Experiment Date']
        runcon['Experiment Date'] = runcon['Experiment Date'].apply(\
            lambda x: pd.to_datetime(str(x),format='%Y%m%d'))
        # print(runcon['Experiment Date'].head(),file=sys.stderr)
        mapping = dict()
        exceptionExps = ['MX1299','MX1269','MX4757','MX4760','MX2343','MX2347','MX2474','MX2477']
        #unpack the runcontents file and store as dictionary by mainKey of SPid
        # followed by secondary key of experiment date.
        for i,(meas,pmap,cytoVedu,expDate,cellLine,mag,timePt) in enumerate(\
            runcon[['Measurement Name', 'Plate Map File','EdU (W2)','Experiment Date',"Cell Lines","Magnification","TimePoint"]].values):
            if meas in exceptionExps:
                continue

            # print('NonExecption,hopefully',file=sys.stderr)
            if pmap not in mapping.keys():
                mapping[pmap] = collections.defaultdict(dict)
                # mapping[pmap][cellLine][mag][timePt] = dict()
            if cytoVedu == 1:
                mapping[pmap][expDate]['EdU'] = list()
                mapping[pmap][expDate]['EdU'].append(\
                    (meas,cytoVedu,i,cellLine,mag.lower(),timePt))
            elif cytoVedu == 0:
                mapping[pmap][expDate]['Cyto'] = list()
                mapping[pmap][expDate]['Cyto'].append(\
                    (meas,cytoVedu,i,cellLine,mag.lower(),timePt))

        self.lastRun.recordLastRunContentsDict(mapping)
        return mapping

    def _extract_releventMXIDs(self,mapping,pmap,exp):
        ''' local function for code deduplication.
            using the mapping dictionary, find the appropriate cyto and EdU
            histdiff files and return the correctly formatted list.
        '''
        print(mapping[pmap][exp],file=sys.stderr)
        if len(mapping[pmap][exp].keys()) ==2 and os.path.isfile(os.path.join(\
            self.lastRun.PlateMapsLoc,'{}.csv'.format(pmap))):
            for cytoPlate in mapping[pmap][exp]['Cyto']:
                cytoExpDetails = cytoPlate[3:]
                #get the cyto histdiff file
                df_cyto_name = "{}.histdiff.csv".format(\
                    cytoPlate[0].strip())
                if int(cytoPlate[0].strip().replace("MX","")[-4:]) < 3200:
                    df_cyto = pd.read_csv(os.path.join(self.path,'backup_cytoOutput',df_cyto_name))
                else:
                    df_cyto = pd.read_csv(os.path.join(self.path,df_cyto_name))
                # check the column names of the file opened and correct
                # for the situation where it is an auto generated name
                # used in HD analysis
                filterKey = cytoPlate[1]
                [df_cyto.rename(columns={\
                    col:self.correctColName(col, filterKey,HD=True)},inplace=True)\
                    for col in df_cyto.columns if not\
                    ((col.upper().endswith('CYTO)') or\
                        col.upper().endswith('CYTO')))]
                try:

                    for eduPlate in mapping[pmap][exp]['EdU']:
                        eduExpDetails = eduPlate[3:]
                        if eduExpDetails == cytoExpDetails:
                            #perform the same with the edu histdiff file
                            df_edu_name = "{}.histdiff.csv".format(\
                                eduPlate[0].strip())
                            if int(eduPlate[0].strip().replace("MX","")[-4:]) < 3200:
                                df_edu = pd.read_csv(os.path.join(self.path,'backup_cytoOutput',df_edu_name))
                            else:
                                df_edu = pd.read_csv(os.path.join(self.path,df_edu_name))

                            filterKey = eduPlate[1]
                            [df_edu.rename(columns={\
                                col:self.correctColName(col, filterKey,HD=True)},inplace=True)\
                                for col in df_edu.columns if not\
                                ((col.upper().endswith('EDU)') or\
                                    col.upper().endswith('EDU')))]

                            # format the two histdiff files in a list ready for actual merge
                            dfs = [(df_edu_name.split(".")[0],df_edu),\
                                (df_cyto_name.split(".")[0],df_cyto),\
                                eduExpDetails]
                            emergencyFill=(cytoPlate[0],eduPlate[0])
                            yield (dfs,";".join([mx for mx in emergencyFill]))
                except KeyError:
                    pass

    def concatPlates(self,force=False):
        ''' workhorse function that will do horizontal concatenation of experiment
         plates '''

        mergeON = 'Features'
        mapping = self._parseRunCon() #digest the provided runcontents file into a dictionary of experiments
        # print(mapping,file=sys.stderr)
        current_exp=tuple()

        for pmap,expDates in mapping.items():
            for exp in expDates:
                for dfs,emergencyFill in self._extract_releventMXIDs(mapping,pmap,exp):
                    cell,mag,timePt = dfs[-1]
                    if force:
                        dfs,emergencyFill = dfs,emergencyFill
                        current_exp = (mapping[pmap][exp]['Cyto'],
                            mapping[pmap][exp]['EdU'], pmap, exp)
                        print('forced new experiment to process',file=sys.stderr)

                    elif not force:
                        #check if this experiment date is later than the last run date
                        #only process if that is true.
                        if not os.path.isfile(os.path.join(\
                            self.lastRun.saveRawPath,\
                            '{}_{}_{}_{}_{}_RAW_histdiff_Concatenated.csv'.\
                            format(pmap,exp.strftime('%Y%m%d'),cell,mag,timePt))):
                            #collect the current experiment details for pickling later
                            #as the lastRun experiment
                            current_exp = (mapping[pmap][exp]['Cyto'],
                                mapping[pmap][exp]['EdU'], pmap, exp)
                            dfs,emergencyFill = dfs, emergencyFill
                            print('new experiment to process',file=sys.stderr)
                        else:
                            dfs = False
                            emergencyFill = False
                            current_exp = False

                    if dfs:
                        #pop off the descriptive file name info associated with the
                        #paired MX files.
                        cell, mag,timePt = dfs.pop()
                        #unpack the first histdiff file and a corresponding name
                        meas1,df1 = dfs.pop()
                        #loop over remaining histdiff files (there really should be only 1)
                        for meas2,df2 in dfs:
                            # print(meas1,meas2)
                            # try:
                            df1 = df1.merge(df2,left_on=mergeON, right_on=mergeON,\
                            how='outer',suffixes=('_'+meas1,'_'+meas2))
                            # except KeyError:
                            #     df1 = df1.merge(df2,left_index=True, right_index=True,\
                            #     how='outer',suffixes=('_'+meas1,'_'+meas2))
                            #     df1['Concentration'] = [float(x.split('_')[-1]) for x \
                            #         in df1.index]

                        df1['PlateName'] = [pmap for _ in range(df1.shape[0])]
                        # print(df1[mergeON].head(),df1.columns,file=sys.stderr)
                        conc = list()
                        for x in df1[mergeON]:
                            concatdName = x.split("_")
                            try:
                                conc.append(float(concatdName[-1]))
                            except ValueError:
                                conc.append(float('-1'))

                        df1['Concentration'] = conc
                        df1['CompoundName'] = [str("_".join(x.split("_")[:-1])) for x \
                            in df1['Features']]
                        df1.fillna(0,inplace=True)
                        # print(df1.axes[1],type(df1))
                        #rename column names to have alphanumeric characters only
                        [df1.rename(index=str,columns={column:self.cleanCNames(column)},\
                            inplace=True) for column in df1.axes[1]]

                        df1.drop(columns='Features',inplace=True)
                        # set index and avoid awkward index/unnamed column at
                        # beginning of plate then save to new CSV file at save location.
                        df1.set_index('CompoundName', inplace=True)
                        df1.drop(index='Blank',inplace=True)
                        df1 = self._wellFinder(df1,pmap,emergencyFill)
                        if df1 is not None:
                            df1.to_csv(os.path.join(\
                                self.lastRun.saveRawPath,\
                                '{}_{}_{}_{}_{}_RAW_histdiff_Concatenated.csv'.\
                                format(pmap,exp.strftime('%Y%m%d'),cell,mag,timePt)))

                            #UNCOMMENT THIS WHEN WE HAVE THE KEEPFEATURES LIST
                            keepColumns_localCopy = self.keepColumns.copy()
                            for col in self.keepColumns:
                                if col not in df1.columns:
                                    keepColumns_localCopy.remove(col)

                            df1= df1[keepColumns_localCopy]
                            df1.to_csv(os.path.join(\
                                self.lastRun.saveCPpath,\
                                '{}_{}_{}_{}_{}_CP_histdiff_Concatenated.csv'.\
                                format(pmap,exp.strftime('%Y%m%d'),cell,mag,timePt)))
        if current_exp:
            #update the pickle object for the last processed plate and save it.
            self.lastRun.recordLastRun(cytoPlate =current_exp[0],eduPlate = current_exp[1],\
                SPplate = current_exp[2], expDate = current_exp[3])
        lastRunPickleObj(update=self.lastRun)

def lastRunPickleObj(update=False):
    '''read and also store file paths for the last run.
        Update=True option will be the way to store the last plateMap processed
        from the CytoRunContentsFile, as well as the digested pmap
    '''
    import pickle
    if os.path.exists(os.path.join(os.path.abspath('.'),"lastRunStainMerge.pkl"))\
        and (update == False):

        stored_paths = pickle.load(open("lastRunStainMerge.pkl","rb"))
        # print(stored_paths.PlateMapsLoc,stored_paths.MXFileLoc,stored_paths.IXMLoc)
        return stored_paths
    elif isinstance(update,tuple):
        stored_paths = LastRun(os.path.abspath(update[0]), #plateMapDir\
            os.path.abspath(update[1]))#,\ #HistDiff MX file location
            #remove closing parenthesis and comments to re-enable IXM file path sortage
            #os.path.abspath(input("IXMFileDictory:")))
        print("storing lastRunStainMerge.pkl",file=sys.stderr)
        pickle.dump(stored_paths,open("lastRunStainMerge.pkl","wb"))
        return stored_paths

    elif isinstance(update,LastRun):
        pickle.dump(update,open('lastRunStainMerge.pkl','wb'))

    else:
        print("ERROR in finding lastRunPickleObj",file=sys.stderr)
        sys.exit(2)


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
