#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributor Paraver trace file (.prv) parsing is a prepossessing operation.
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 14/12/2018

import argparse
import multiprocessing
import os.path
import subprocess
import sys
import time
import warnings
import shutil
import h5py
import csv
import re
import pandas as pd
import kavica.parser.prvparse as prvparse
import pickle


class Distributor(object):
    """ It is shrink the data file and distribute it among the processes for parsing."""

    def __init__(self, stop=None):
        self._multi_threads = 1
        self._multi_process = 1
        self.__chunks = None
        self.__chunk_size = None
        self._path = None
        self.__isDistributed = False
        self.stop = stop
        # TODO: those have to initiated and read from the configuration file or retrieval from meta data processing.
        self.root_dir = '../temp'  # Unique chunk directory
        self.filepath = '../temp/fragment'
        self._output_path = os.path.join('output.csv')
        self.hdfpath = os.path.join('source.hdf5')
        self.csvpath = os.path.join('source.csv')
        self.__csv_chunks = []
        self.csv_header = []

    def __call__(self, *args, **kwargs):
        self.arguments_parser()
        if self.__isDistributed:
            self.shater()
        self._parse()

    def arguments_parser(self):
        # set/receive the arguments
        if len(sys.argv) == 1:
            arguments = ['config/config_gromacs_64p_L12-CYC.json',
                         '../../data/gromacs_64p.L12-CYC.prv',
                         '-o',
                         'output.csv',
                         # '-mt',
                         # '5',
                         '-mp',
                         '10'
                         ]
            sys.argv.extend(arguments)
        else:
            pass

        # parse the arguments
        parser = argparse.ArgumentParser(description='The files that are needed for parsing.')
        parser.add_argument('config', help='A .json configuration file that included the'
                                           'thread numbers,hardware counters and etc.')
        parser.add_argument('prvfile', help='A .prv trace file')
        parser.add_argument('-c',
                            dest='c',
                            action='store',
                            type=str,
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
        self._path = args.prvfile

        if args.o:
            self._output_path = args.o

        if args.mt:
            if args.mt < 2:
                raise ValueError("Multi_Thread number has to be (=> 2). It is set {}".format(args.mt))
            else:
                self._multi_threads = args.mt
                self.__isDistributed = True

        if args.mp:
            if args.mp < 2:
                raise ValueError("Multi_Possess number has to be (=> 2). It is set {}".format(args.mp))
            else:
                self._multi_process = args.mp
                self.__isDistributed = True
                if self._multi_process > multiprocessing.cpu_count():
                    warnings.warn('{} processes are available, but {} is requested.'.format(multiprocessing.cpu_count(),
                                                                                            self._multi_process),
                                  UserWarning)

    def shater(self):
        """ It is shattered the .prv/text file among the processes/threads."""
        # Todo: run the Paraver filter

        # read the number of lines
        line_numbers_command = "wc -l < " + self._path
        lines_number = int(subprocess.check_output(line_numbers_command, shell=True))
        self.__chunks = self._multi_process * self._multi_threads
        self.__chunk_size = round(lines_number / self.__chunks)

        if self.stop is None:
            self.stop = self.__chunk_size
        else:
            pass

        print("The parser is run over {} processes , {} lines per process and stop after parsing {} lines.".format(
            self.__chunks, self.__chunk_size, self.stop))

        # Clear previous results
        # Todo: the last line of any chunk has to be repeated to the next one.
        if os.path.isdir(self.root_dir):
            warnings.warn('The temporal directory already exist. I will be deleted and recreated by parser.')
            shutil.rmtree(self.root_dir)
        if os.path.isfile(self.hdfpath):
            os.remove(self.hdfpath)
        if os.path.isfile(self.csvpath):
            os.remove(self.csvpath)

        # Create new directory
        os.makedirs(self.root_dir)

        # Create chunked in many .prv files
        split_command = 'split -l {} --numeric-suffixes --additional-suffix=.prv {} {}'.format(
            self.__chunk_size,
            self._path,
            self.filepath)
        subprocess.check_output(split_command, shell=True)

        # Create the HDF5 file for combining the parsed data.
        with h5py.File(self.hdfpath) as hdf5_source:
            hdf5_source.create_group('chunks')

        # read and write the meta-data to the hdf5
        # TODO: Write the meta_data reader

    def _parse(self):
        """ It tunes multi processes and runs the parsers in any of them."""

        def __update_output_file():
            """ Integrate the data in to one .csv/.hdf5 file."""
            # Obtain the header
            with open('csvh.pkl', 'rb') as pkl_header:
                self.csv_header = pickle.load(pkl_header)

            # Write the output as hdf5 & CSV dataset
            with open(self.csvpath, mode='w') as outputCSV:
                output_csv_writer = csv.writer(outputCSV)
                output_csv_writer.writerow(self.csv_header)  # Insert the header in to the .csv file.
                with h5py.File(self.hdfpath, 'w') as hdf5_source:
                    # TODO: add header to HDF5 file
                    hdf5_data_type = h5py.special_dtype(vlen=str)
                    # TODO: add meta data about the file to the HDF5
                    for csv_file in self.__csv_chunks:
                        chunk_data = pd.read_csv(str(self.root_dir + '/' + csv_file + '.csv'),
                                                 index_col=None,
                                                 header=None)
                        # Write to the HDF5
                        hdf5_source.create_dataset(csv_file, data=chunk_data, dtype=hdf5_data_type)

                        # Write to the CSV
                        output_csv_writer.writerows(chunk_data.values)

                        # How to extract the dataset data from hdf5 ->data = hdf5_source[csv_file].value
            # Test the final csv shape.
            chunk_data = pd.read_csv(self.csvpath, index_col=None, header=None)
            print(chunk_data.shape)

            # TODO: Delete the temporal files
            if os.path.isdir(self.root_dir):
                warnings.warn('The /temp/ directory will have been not needed. It has been deleted.')
                shutil.rmtree(self.root_dir)


        def __chunk_pars(chunk_name, parser_object):
            """ Setup the arguments and run the parser objects"""
            sys.argv[2] = str(self.root_dir) + '/' + str(chunk_name)
            parser_object(stop=self.stop)

        # Handling the different distribution circumstances.
        if not self.__isDistributed:  # < 1 thread, 1 processes >
            warnings.warn('The parsing is not distributed, It will executed by one possessor/thread!!!', UserWarning)
            args = prvparse.Parser()
            args(stop=self.stop)
            # Test the final csv shape.
            chunk_data = pd.read_csv(self._output_path, index_col=None, header=None)
            print(chunk_data.shape)

        elif self._multi_process > 1:
            chunk_list = filter(lambda x: x.startswith('fragment'), os.listdir(self.root_dir))
            if self._multi_threads == 1:  # < 1 thread, Multi processes >
                parser_list = [prvparse.Parser(chunk_id=chunk) for chunk in range(self._multi_process)]
                self.__csv_chunks = [re.sub(r'\W', '', str(csv_chunk).split()[-1]) for csv_chunk in
                                     parser_list]

                processes = [multiprocessing.Process(target=__chunk_pars, args=(chunk, parser_object,)) for
                             chunk, parser_object
                             in zip(chunk_list, parser_list)]

                # Run processes
                for p in processes:
                    p.start()

                # Exit the completed processes
                for p in processes:
                    p.join()
                    if p.is_alive():
                        print("Job {} is not finished!".format(p))
                __update_output_file()

            else:  # < Multi thread, Multi processes >
                raise EnvironmentError(
                    "The multi threading has not been available yet. Try <-mp {}> instead of <-mp {} -mt {}>".format(
                        self._multi_threads * self._multi_process,
                        self._multi_process,
                        self._multi_threads))
        else:  # < Multi thread, 1 processes >
            raise EnvironmentError(
                "The multi threading has not been available yet. Try <-mp {}> instead of <-mt {}>".format(
                    self._multi_threads,
                    self._multi_threads))


def distributing():
    start = time.time()
    try:
        # Developtime: In order to reduce the test time, Distributor(stop=3496) should be used.
        distributed = Distributor()
        distributed()
        print("\033[32mThe trace file {} is parsed successfully to both .hdf5 and .csv file.".format(distributed._path))
    except:
        print("\033[31mThe parsing proses is failed.")
    finally:
        duration = time.time() - start
        print('\033[0mTotal duration is: %.3f' % duration)


if __name__ == '__main__':
    distributing()
