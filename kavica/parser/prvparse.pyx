#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paraver trace file (.prv) parsing is a prepossessing operation.

Compile and build:
    $ python3 setup.py build_ext --inplace
    $ mv kavica/parser/prvparse.cpython-36m-x86_64-linux-gnu.so .

    Fixme: revise the setup_factory for the whole package
        - python3 setup_factory.py build_ext --inplace
        - mv kavica/parser/prvparse.cpython-36m-x86_64-linux-gnu.so .

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 27/09/2022
"""

from terminaltables import DoubleTable
from argparse import ArgumentTypeError
import os
import json
import sys
import pandas as pd
import numpy as np
import psutil
import warnings
import gc
import argparse
import csv
import itertools
import signal
import re
import pickle
import networkx as nx
cimport numpy as np


#TODO: add applicability of filters to reduce the size of trace
#TODO: in case of breaking the parsing in the middle, it is needed to save the data in output file
#TODO: Split and distribute the file HDF5 among the MPI parser

class ControlCZInterruptHandler(object):

    def __init__(self, signals=(signal.SIGINT, signal.SIGTSTP), reignOfCode=None):
        self.reignOfCode = reignOfCode
        self.signals = signals
        self.original_handlers = {}
        self.interrupted = False
        self.released = False

    def __enter__(self):

        def handler(signum, frame):
            if self.reignOfCode == 'Parser':
                if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                    self.release()
                    self.interrupted = True
                else:
                    print('continue')
            else:
                print('continue again')

        for sig in self.signals:
            self.original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

        return self

    def handler(self, signum, frame):
        self.release()
        self.interrupted = True

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        for sig in self.signals:
            signal.signal(sig, self.original_handlers[sig])

        self.released = True
        return True


cdef class ExtensionPathType(object):
    cdef:
        bint _exists
        str _type
        bint _dash_ok
        str error

    def __init__(self, exists=True, types='file', dash_ok=True):
        """exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
           type: file, dir, symlink, None, or a function returning True for valid paths
                None: don't care
           dash_ok: whether to allow "-" as stdin/stdout
        """

        assert exists in (True, False, None)
        assert types in ('file', 'dir', 'symlink', None) or callable(type)

        self._exists = exists
        self._type = types
        self._dash_ok = dash_ok
        self.error = ''

    def __call__(self, string, extension, arg=None):
        string = str(string)
        if string == '-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise ArgumentTypeError('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise ArgumentTypeError('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise ArgumentTypeError('standard input/output (-) not allowed')
            return string  # No reason to check anything else if this works.

        exists = os.path.exists(string)
        if self._exists:
            if not exists:
                if arg == 'o':
                    warnings.warn("The output file is not exist.", UserWarning)
                    createOutput = 'y'  #input("Would you like to create it? y/n")
                    if createOutput == 'y':
                        with open('output.csv', mode='w'):
                            pass
                    else:
                        warnings.warn("The output is not saved", UserWarning)
                        return
                else:
                    raise ArgumentTypeError("path does not exist: '%s'" % string)
            if self._type is None:
                pass
            elif self._type == 'file':
                if not os.path.isfile(string):
                    raise ArgumentTypeError("path is not a file: '%s'" % string)
            elif self._type == 'symlink':
                if not os.path.islink(string):
                    raise ArgumentTypeError("path is not a symlink: '%s'" % string)
            elif self._type == 'dir':
                if not os.path.isdir(string):
                    raise ArgumentTypeError("path is not a directory: '%s'" % string)
            elif not self._types(string):
                raise ArgumentTypeError("path not valid: '%s'" % string)
            if not os.path.splitext(string)[1] == extension:
                raise ArgumentTypeError("file is not %s: '%s'." % (extension, string))
        else:
            if not self._exists:
                if exists:
                    raise ArgumentTypeError("path exists: '%s'" % string)
            p = os.path.dirname(os.path.normpath(string)) or '.'
            if not os.path.isdir(p):
                raise ArgumentTypeError("parent path is not a directory: %r" % p)
            elif not os.path.exists(p):
                raise ArgumentTypeError("parent directory does not exist: %r" % p)
        return string

cdef class ParsedArgs(ExtensionPathType):
    cdef:
        str _config
        str _prvfile
        str _output
        bint _clustered
        bint _is_distributed

    def __init__(self, config=None, prvfile=None, outputfile=None, clustered=False):
        super(ParsedArgs, self).__init__()
        self._config = config
        self._prvfile = prvfile
        self._output = outputfile
        self._clustered = clustered
        self._is_distributed = False

    def __call__(self):

        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        parser = argparse.ArgumentParser(description='The files that are needed for parsing.')
        parser.add_argument('config', help='A .json configuration file that included the'
                                           'thread numbers,hardware counters and etc.')
        parser.add_argument('prvfile', help='A .prv trace file')
        parser.add_argument('-c',
                            dest='c',
                            action='store',
                            type=str2bool,
                            help="True, if the .prv trace file has class label.")
        parser.add_argument('-o',
                            dest='o',
                            action='store',
                            type=str,
                            help="path to custom root results directory")

        parser.add_argument('-mp',
                            dest='mp',
                            type=int,
                            help="Apply Multi_Possessing. It has to be => 2.")
        parser.add_argument('-mt',
                            dest='mt',
                            type=int,
                            help="Apply Multi_Threading.It has to be => 2.")

        args = parser.parse_args()
        self._config = super(ParsedArgs, self).__call__(args.config, '.json')
        self._prvfile = super(ParsedArgs, self).__call__(args.prvfile, '.prv')
        self._clustered = args.c
        if args.o:
            self._output = super(ParsedArgs, self).__call__(args.o, '.csv', arg='o')

        if args.mt or args.mp:
            self._is_distributed = True

cdef class Parser(ParsedArgs):
    cdef:
        dict hardware_counters
        int thread_numbers
        dict call_stacks
        list header
        np.ndarray stateRecord
        np.ndarray eventRecord
        np.ndarray communicationRecord
        list stateRecordHeader
        list eventRecordHeader
        list communicationRecordHeader
        list csv_header
        int chunk_id
        dict last_time
        list events_indicators
        str  root_dir


    def __init__(self, chunk_id=1):
        super(Parser, self).__init__()
        self.chunk_id = chunk_id
        self.root_dir = '../temp'
        with open('{}/events_index_list.pkl'.format(self.root_dir), 'rb') as f:
            events_indicators = pickle.load(f)
        self.events_indicators = events_indicators

    def __cinit__(self, hardware_counters={}, thread_numbers=0, clustered=False, call_stacks={}, chunk_id=1):
        super(Parser, self).__init__()
        self.chunk_id = chunk_id
        self.hardware_counters = hardware_counters
        self.thread_numbers = thread_numbers
        self.call_stacks = call_stacks
        self.last_time = None
        self.header = []
        self.csv_header = []
        self.stateRecord = np.empty([0, 4], dtype=np.str)
        self.eventRecord = np.empty([0, 0], dtype=np.str)
        self.communicationRecord = np.empty([0, 8], dtype=np.str)
        self.stateRecordHeader = ['Object_id',
                                  'begin_time',
                                  'end_time',
                                  'state']
        self.eventRecordHeader = ['Object_id',
                                  'Timestamp',
                                  'Duration']  # HC are added by configuration parser
        self.communicationRecordHeader = ['Object_send',
                                          'lsend',
                                          'psend',
                                          'Object_recv',
                                          'lrecv',
                                          'precv',
                                          'size',
                                          'tag']

    def __call__(self, stop=None):
        super(Parser, self).__call__()
        self.parse_config()
        self.parse_pfc()
        self.parse_prv(stop=stop)
        self.csv_headline()
        for i in self.communicationRecord:
            print(i)
        return self.csv_header

    @staticmethod
    def progress_bar(counter, total, process_id, status=''):
        bar_len = 40
        filled_len = int(round(bar_len * counter / float(total)))
        percents = round(100.0 * counter / float(total), 1)
        bar = '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('\r\033[1;36;m[%s] chunk_id <%s> %s%s ...%s' % (bar, process_id, percents, '%', status))
        sys.stdout.flush()
        return 0

    @classmethod
    def memory_status(cls, initMemoryStatus, finalMemoryStatus):
        collected = gc.collect()
        print ("\n Garbage collector: collected %d objects." % (collected))
        collectedGarbageMemoryStatus = psutil.virtual_memory()  # final memory status is recorded
        initMemoryStatus = iter(initMemoryStatus)
        finalMemoryStatus = iter(finalMemoryStatus)
        collectedGarbageMemoryStatus = iter(collectedGarbageMemoryStatus)

        tableData = [['', 'Initial', 'Final', 'GC']]
        memoryParametersList = ['total',
                                'available',
                                'percent',
                                'used',
                                'free',
                                'action',
                                'inavtion',
                                'buffer',
                                'cached',
                                'shared',
                                'slab']

        for [memoryItem, initStatus, finalStatus, collectedGarbageStatus] in zip(memoryParametersList,
                                                                                 initMemoryStatus,
                                                                                 finalMemoryStatus,
                                                                                 collectedGarbageMemoryStatus):
            tableData.append([memoryItem, initStatus, finalStatus, collectedGarbageStatus])
        table = DoubleTable(tableData)
        #Developtime: It is used for checking the memory status.
        #print (table.table)

    def csv_headline(self):
        with open('{}/csvh.pkl'.format(self.root_dir), 'wb') as header_pkl:
            pickle.dump(self.csv_header, header_pkl)

    def outputCSV(self, record_type='event'):
        assert record_type in ('event', 'status', 'communication')
        if self._clustered:
            headerFile = list(itertools.chain.from_iterable([self.eventRecordHeader,
                                                             self.hardware_counters.values(),
                                                             self.call_stacks.values(),
                                                             ['label']]))
        else:
            headerFile = list(itertools.chain.from_iterable([self.eventRecordHeader,
                                                             self.hardware_counters.values(),
                                                             self.call_stacks.values()]))
        self.csv_header = headerFile

        if self._is_distributed:
            # Chunked CS
            chunk_name = re.sub(r'\W', '', str(self).split()[-1]) + '.csv'
            with open(str('../temp/' + chunk_name), mode='w') as outputCSV:
                outputCsvWriter = csv.writer(outputCSV)
                outputCsvWriter.writerows(self.eventRecord)
        else:
            # copulative CSV
            with open(str(self._output), mode='w') as outputCSV:
                outputCsvWriter = csv.writer(outputCSV)
                outputCsvWriter.writerow(headerFile)
                outputCsvWriter.writerows(self.eventRecord)

    def set_headers(self, injectHeader=False):
        """Set the column heater as first row of any numpy array that will be used
         during converting to the DataFrame"""
        if self._clustered:
            self.eventRecordHeader = self.eventRecordHeader + \
                                     list(self.hardware_counters.values()) + \
                                     list(self.call_stacks.values() + \
                                          list(['label']))
        else:
            self.eventRecordHeader = self.eventRecordHeader + \
                                     list(self.hardware_counters.values()) + \
                                     list(self.call_stacks.values())

        if injectHeader:
            self.stateRecord = np.vstack((self.stateRecord, self.stateRecordHeader))
            self.eventRecord = np.vstack((self.eventRecord, self.eventRecordHeader))
            self.communicationRecordHeader = np.vstack((self.communicationRecord,
                                                        self.communicationRecordHeader))

    def to_dataframe(self, numpyData=None):
        numpyData = numpyData if numpyData is not None else self.eventRecord  # set the default
        datafrmeData = pd.DataFrame(data=numpyData[1:, 1:],
                                    index=numpyData[1:, 0],
                                    columns=numpyData[0, 1:], )
        print(datafrmeData)
        return datafrmeData

    def parse_config(self):
        with open(self._config) as config:
            try:
                config_items = json.load(config)
                self.hardware_counters = config_items["hardware_counters"]
                self.thread_numbers = config_items["thread_numbers"]
                if self._clustered is not None:
                    if self._clustered == config_items["clustered"]:
                        pass
                    else:
                        #fixme: it dose not work with input-> clustered Dubel Check = input('Configuration and command line argument conflict.Is it clustered? y/n: ')
                        clusteredDubelCheck = 'y'
                        if clusteredDubelCheck.lower() == 'y':
                            self._clustered = True
                        elif clusteredDubelCheck.lower() == 'n':
                            self._clustered = False
                else:
                    self._clustered = config_items["clustered"]
                self.call_stacks = config_items["call_stacks"]

                # reshape the main data array based on the number of hardware counters
                column_number = int(len(self.eventRecordHeader) +
                                    len(self.hardware_counters) +
                                    len(self.call_stacks))
                if self._clustered:
                    column_number += 1
                else:
                    pass

                self.eventRecord = np.reshape(self.eventRecord, (0, column_number))

                # initiate the headers
                #self.set_headers(True)

            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
                raise ArgumentTypeError("Could not read file: %s" % self._config)
            except:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
                raise ArgumentTypeError("Could not read file: %s" % self._config)

    cdef hc_validator(self, hcList):
        cdef:
            dict hcDict
            dict callStackDict
            str hcItem = ''
            float IPC = 0

        # Define the primary dictionaries
        hcDict = self.hardware_counters.fromkeys(self.hardware_counters, None)
        callStackDict = self.call_stacks.fromkeys(self.call_stacks, None)
        hcList = iter(hcList)
        for hcItem in hcList:
            if hcItem in self.hardware_counters.keys():
                hcDict.update({hcItem: next(hcList)})
            elif hcItem in self.call_stacks.keys():
                callStackDict.update({hcItem: next(hcList)})

        # convert the values to integer
        hcDict = {k: int(0 if v is None else v) for k, v in hcDict.items()}
        callStackDict = {k: int(0 if v is None else v) for k, v in callStackDict.items()}

        # TODO: read more formula from configuration file and generate more derived hardware counters.
        # calculating the IPC
        TOT_INS = hcDict.get('42000050')
        TOT_CYC = hcDict.get('42000059')
        if TOT_CYC != 0:
            hcDict.update({'IPC': float(TOT_INS / TOT_CYC)})
        return {'hcDict': hcDict, 'callStackDict': callStackDict}

    # TODO: It will generate the meta data about the trace file. It reads the .pfc, .row and .prv header.
    def parse_pfc(self, thread_per_core=None):
        # TODO: i should be updated based on the real thread numbering, it just working for thread per core.
        if thread_per_core == 1:
            thread_list = ["{0}.{1}.{2}.{3}".format(thread_item,
                                                    thread_per_core,
                                                    thread_item,
                                                    thread_per_core) for thread_item in
                           range(1, self.thread_numbers + 1)]
        else:
            thread_list = self.events_indicators
        init_thread_timestamp = dict.fromkeys(thread_list, 0)
        self.last_time = dict.fromkeys(['state', 'event', 'communication'], init_thread_timestamp)

    def parse_prv(self, inMemory=True, stop=200000):

        cdef:
            int totalLinesNumber = 0
            list newRow
            char*c_string = ''
            object traceLines = c_string
            dict hcValidatedDict

        def parsing(traceLinesSet, totalLinesNumber, clustered=False):
            clusterTemp = [None, None, None]
            lineConter = 1
            for traceLines in traceLinesSet:
                traceLines = traceLines.replace("\n", "")
                Parser.progress_bar(lineConter, totalLinesNumber, self.chunk_id,
                                    status=str(lineConter) + "/" + str(totalLinesNumber))
                newRow = []
                if traceLines[0] == '1':  # state data
                    traceLines = list(traceLines.split(':'))
                    newRow.append("{0}.{1}.{2}.{3}".format(traceLines[1],
                                                           traceLines[2],
                                                           traceLines[3],
                                                           traceLines[4]))
                    newRow += [traceLines[5],
                               traceLines[6],
                               traceLines[7]]
                    self.stateRecord = np.vstack((self.stateRecord, newRow))

                elif traceLines[0] == '2':  # events data
                    traceLines = list(traceLines.split(':'))
                    thread_id = ("{0}.{1}.{2}.{3}".format(traceLines[1],
                                                          traceLines[2],
                                                          traceLines[3],
                                                          traceLines[4]))
                    newRow.append(thread_id)

                    duration = int(traceLines[5]) - int(self.last_time['event'][thread_id])
                    self.last_time['event'][thread_id] = traceLines[5]
                    newRow += [traceLines[5], duration]

                    if self._clustered:
                        if traceLines[6] == '90000001':
                            newRow += [traceLines[7]]
                            clusterTemp = newRow
                        else:
                            hcValidatedDict = self.hc_validator(traceLines[6:])
                            newRow += hcValidatedDict['hcDict'].values()
                            newRow += hcValidatedDict['callStackDict'].values()
                            newRow += [str(clusterTemp[-1])]
                            clusterTemp = [None, None, None]
                            self.eventRecord = np.vstack((self.eventRecord, newRow))
                    else:
                        hcValidatedDict = self.hc_validator(traceLines[6:])
                        newRow += hcValidatedDict['hcDict'].values()
                        newRow += hcValidatedDict['callStackDict'].values()
                        self.eventRecord = np.vstack((self.eventRecord, newRow))

                elif traceLines[0] == '3':  # communication
                    traceLines = list(traceLines.split(':'))
                    thread_id = ("{0}.{1}.{2}.{3}".format(traceLines[1],
                                                          traceLines[2],
                                                          traceLines[3],
                                                          traceLines[4]))
                    newRow.append(thread_id)
                    newRow += [traceLines[5],
                               traceLines[6]]
                    newRow.append("{0}.{1}.{2}.{3}".format(traceLines[7],
                                                           traceLines[8],
                                                           traceLines[9],
                                                           traceLines[10]))
                    newRow += [traceLines[11],
                               traceLines[12],
                               traceLines[13],
                               traceLines[14]]
                    self.communicationRecord = np.vstack((self.communicationRecord, newRow))
                else:  # Header
                    self.header.append(traceLines)
                lineConter += 1
                if lineConter == stop:
                    break

        with ControlCZInterruptHandler(reignOfCode=self.__class__.__name__) as ControlCZ:
            if not ControlCZ.interrupted:
                initMemoryStatus = psutil.virtual_memory()  # final memory status is recorded

                with open(self._prvfile, 'r') as traceFile:
                    traceLinesSet = traceFile.readlines()
                    try:
                        if not inMemory:
                            with open(self._prvfile, 'r') as traceFile:  # extract the number of rows
                                totalLinesNumber = len(traceFile.readlines())
                                traceFile.close()
                        else:
                            totalLinesNumber = len(traceLinesSet)
                        parsing(traceLinesSet, totalLinesNumber)
                    except IOError as e:
                        print("I/O error({0}): {1}".format(e.errno, e.strerror))
                        raise ArgumentTypeError("Could not read file: %s" % self._prvfile)

                    except:  # handle other exceptions such as attribute errors
                        print("Unexpected error:", sys.exc_info()[0])
                        raise ArgumentTypeError("Could not read file: %s" % self._prvfile)

                # Fixme: improve the memory statue report
                finalMemoryStatus = psutil.virtual_memory()  # final memory status is recorded
                #self.memory_status(initMemoryStatus, finalMemoryStatus)
                self.outputCSV()
            else:
                #TODO: stop the parser and save the data partially
                pass

cdef class ProgramActivityGraphConstructor(object):
    """ Construct the Program Activity Graph from timestamped trace file.
    It generates a directed graph when:

    Activities on vertexes:
    The vertexes are either CPU burst,message-passing, transmission of messages
    and overhead associated with asynchronously receiving messages with duration
    and the edges represent the predecessor relation between activities.

    Activities on edges:
    The edges are either CPU burst,message-passing, transmission of messages
    and overhead associated with asynchronously receiving messages with duration
    and the vertexes timestamped events.

    Parameters
    ----------
    Name : dict, optional

    Attributes
    ----------
    Name : type

    Notes
    -----


    --------
    kavica.graph_data_structur : convenience data structure for
        unifying the pic.

    Examples
    --------

    """

    cdef:
        object PAG
        str chunk_id
        str temp_dir
        str return_key
        str overall_finish_timestamp

    def __init__(self, chunk_id=1, temp_dir=None, overall_finish_timestamp=0):
        self.chunk_id = chunk_id
        self.temp_dir = temp_dir
        self.return_key = self.chunk_id.replace("fragment_", "").replace(".txt", "")
        self.overall_finish_timestamp = overall_finish_timestamp

    def __cinit__(self):
        self.PAG = nx.Graph()

    def __call__(self, return_dict):
        self.__pag_constructor(return_dict)

    @staticmethod
    def progress_bar(counter, total, process_id, status=''):
        bar_len = 40
        filled_len = int(round(bar_len * counter / float(total)))
        percents = round(100.0 * counter / float(total), 1)
        bar = '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('\r\033[1;36;m[%s] chunk_id <%s> %s%s ...%s' % (bar, process_id, percents, '%', status))
        sys.stdout.flush()
        return 0

    def __pag_constructor(self, return_dict):
        # data read and preparation.
        chunk_path = str('{}/{}'.format(self.temp_dir, self.chunk_id))
        data = pd.read_csv(chunk_path, sep=":", header=None)

        def construct_status_pag():
            """
            Parse the status events of sequential tasks usually for any thread.
            Where, all the nodes at lest has one predecessor and one successor.
            :return: networkx graph where weight is duration(length) and color is status
            """
            data['duration'] = data[6] - data[5]

            start_timestamp = data.iloc[0,][5]  # temporal duration of the task initiation stage
            finish_timestamp = data.iloc[-1,][6]  # temporal finish timestamp of the task finishing stage

            data[data.columns[1:7]] = data[data.columns[1:7]].astype(str)
            data['source'] = data[1] + data[2] + data[3] + data[4] + ':' + data[5]
            data['target'] = data[1] + data[2] + data[3] + data[4] + ':' + data[6]
            data.drop(data.columns[0:7], axis=1, inplace=True)
            data.set_axis(['status', 'weight', 'source', 'target'], axis='columns', inplace=True)

            # the index is needed to save before inserting the new records
            head_record = data.head(1).reset_index()
            tail_record = data.tail(1).reset_index()
            # TODO: Adding wait activities as dummy activities
            # TODO: check, is it correct to add duration to the finishing edge
            # add the start and finish nodes [begin,end]
            data.loc['start'] = {'status': 98, 'weight': start_timestamp, 'source': 'start:0',
                                 'target': head_record.ix[0, 'source']}
            data.loc['finish'] = {'status': 99,
                                  'weight': int(self.overall_finish_timestamp) - finish_timestamp,
                                  'source': tail_record.ix[0, 'target'],
                                  'target': str('finish:' + str(self.overall_finish_timestamp))}

            return nx.from_pandas_edgelist(data, edge_attr=True, create_using=nx.DiGraph())

        def construct_communication_pag():
            """
            Parse the communication events usually for any thread.
            :return: networkx graph, where weight is duration(length) and color is tag.
            """
            # data read and preparation.
            data['duration'] = data[12] - data[6]
            data[data.columns[1:7]] = data[data.columns[1:7]].astype(str)
            data[data.columns[7:13]] = data[data.columns[7:13]].astype(str)
            data['source'] = data[1] + data[2] + data[3] + data[4] + ':' + data[6]
            data['target'] = data[7] + data[8] + data[9] + data[10] + ':' + data[12]
            data.drop(data.columns[0:7], axis=1, inplace=True)
            data.drop(data.columns[0:6], axis=1, inplace=True)
            data.set_axis(['size', 'tag', 'weight', 'source', 'target'], axis='columns', inplace=True)

            return nx.from_pandas_edgelist(data, edge_attr=True, create_using=nx.DiGraph())

        def construct_cluster_attribute_pag():
            """
            Parse the cluster label of sequential computation usually for any thread.
            Where, all the nodes at lest has one predecessor and one successor.
            :return: networkx graph where weight is duration(length) and color is status
            """
            data[data.columns[1:7]] = data[data.columns[1:7]].astype(str)
            data[data.columns[7:13]] = data[data.columns[7:13]].astype(str)
            data['source'] = data[1] + data[2] + data[3] + data[4] + ':' + data[5]
            data['target'] = data[1] + data[2] + data[3] + data[4] + ':' + data[5]
            data['target'] = data['target'].shift(-1)
            data.drop(data.columns[0:7], axis=1, inplace=True)
            data.set_axis(['cluster_id', 'source', 'target'], axis='columns', inplace=True)
            # TODO: Adding wait activities as dummy activities
            # TODO: check, is it correct to add duration to the finishing edge
            # add the start and finish nodes [begin,end]
            if data["cluster_id"].iloc[-1] == '6':
                data["cluster_id"].iloc[-1] = 99
            data["target"].iloc[-1] = str('finish:' + str(self.overall_finish_timestamp))
            data.loc['start'] = {'cluster_id': 98, 'source': 'start:0',
                                 'target': data.ix[0, 'source']}

            return nx.from_pandas_edgelist(data, edge_attr=True, create_using=nx.DiGraph())

        def construct_hardware_counter_attribute_pag():
            # todo: pars the HC {fragment_2}
            pass

        if self.chunk_id.startswith('fragment_1'):
            G = construct_status_pag()
        elif self.chunk_id.startswith('fragment_3'):
            G = construct_communication_pag()
        elif self.chunk_id.startswith('fragment_L'):
            G = construct_cluster_attribute_pag()

        return_dict[self.return_key] = G

        return return_dict

