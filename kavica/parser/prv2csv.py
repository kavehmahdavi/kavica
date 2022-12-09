#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Base class to parse .prv trace file to the .csv

Example:
    $ python prv2csv.py config/config_gromacs_64p.json ../../data/parser_test_data/gromacs_64p.chop1.dev.test_configs.prv -mp 12

Note:
    The configuration sample files are available in .config/

Dependency:
    $ cd kavica_container
    $ python3 setup.py install                        or $pip install -e kavica

Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
last update: 23/09/2022
"""
# Fixme: pandas ix replace with loc/iloc
import argparse
import multiprocessing
import os.path
import subprocess
import sys
import warnings
import shutil
import csv
import re
import pandas as pd
import pickle
from kavica.parser import prvparse as prvparse
from kavica.utils import BColors

__all__ = ['Distributor']


class Distributor(object):
    """ Base class to multithread/multiprocess parser

    It is shrink the data file and distribute it among the processes for parsing.

    """

    def __init__(self, stop=None):
        # Set parameters
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
        self.csvpath = os.path.join('source.csv')
        self.__csv_chunks = []
        self.csv_header = []
        self.events_index_list = []

    def __call__(self, *args, **kwargs):
        self.arguments_parser()
        if self.__isDistributed:
            self.shatter()
        self._parse()

    def arguments_parser(self):
        """ Receive the parameters.

        Return:
            Self, it is applicable as a chain
        """
        # DevelopTime: Test parameters
        if len(sys.argv) == 1:
            arguments = [
                './config_gromacs_64p_L12-CYC.json',
                '../data/gromacs_64p.L12-CYC.prv',
                '-o',
                'output.csv',
                # '-mt',
                # '5',
                '-mp',
                '100'
            ]
            sys.argv.extend(arguments)

        # Parse the arguments
        parser = argparse.ArgumentParser(description='The files that are needed for parsing.')
        parser.add_argument('config', help='A .json config includes the thread numbers,hardware counters and etc.')
        parser.add_argument('prvfile', help='A .prv trace file')
        parser.add_argument('-c', dest='c', action='store', type=str, help="True, if the .prv file has class label.")
        parser.add_argument('-o', dest='o', action='store', type=str, help="path to  results directory")
        parser.add_argument('-mp', dest='mp', type=int, help="Apply Multi_Possessing. It has to be => 2.")
        parser.add_argument('-mt', dest='mt', type=int, help="Apply Multi_Threading.It has to be => 2.")
        args = parser.parse_args()
        self._path = args.prvfile

        if args.o:
            self.csvpath = args.o
        if args.mt:
            if args.mt < 2:
                raise ValueError("Multithread: {} (=> 2)".format(args.mt))
            else:
                self._multi_threads = args.mt
                self.__isDistributed = True
        if args.mp:
            if args.mp < 2:
                raise ValueError("Multiprocess: {} (=> 2)".format(args.mp))
            else:
                self._multi_process = args.mp
                self.__isDistributed = True
                if self._multi_process > multiprocessing.cpu_count():
                    warnings.warn('CPU: Available: {}, Requested: {}'.format(multiprocessing.cpu_count(),
                                                                             self._multi_process), UserWarning)
        return self

    # TODO: it needed to added to the prv2gap.py file
    # TODO: add the parameters list to the configuration json.
    def __prv_cleaner(self, line_beginning_character=['c', '#'], not_needed_computation=['1000', '1001', '50000001'],
                      needed_info={'status': True, 'communication': True, 'computation': True}, clustered=False):

        """Clean and filter the data

        Perform the .prv trace file cleaning (Remove the header and comments) before the parsing to the PAG.

        Args:
            line_beginning_character (list of char): The lines are started with an of those char that will be
                                                     eliminated from returned file.

            needed_info (binary dict): key includes all type of the records in the .PRV and value is True if it is
                                       needed to exist in return file else False.

            not_needed_computation (binary dict): key includes all computation in the .PRV and value is True if
                                                  it is not needed to exist in return file else False.

            clustered (boolean): If it is True that indicates the .prv file has clustering information.

        Return:
            Self, it is applicable as a chain

        Raises:
            ValueError: An error occurred passing incorrect information item.
        """

        assert set(needed_info.keys()) <= set(['status', 'communication', 'computation']), \
            " //Error: the info is not available, just 'status', 'communication', 'computation' are accepted."

        # Update the eliminating list with the needed information
        if clustered:
            # change the first column to 'L' (label) for clustering info.
            clustered_cmd = str(
                "awk -F: '($1 ==" + '"2" &&' +
                ' $7 =="90000001")' +
                ' {$1="L"}1' + "' OFS=':' {} > {}/temp_cleaned.prv".format(
                    self._path,
                    self.root_dir))
        else:
            clustered_cmd = str("awk -F: '!($1=='2' && $7 == '90000001')' {} > {}/temp_cleaned.prv".format(self._path,
                                                                                                           self.root_dir))

        subprocess.check_output(clustered_cmd, shell=True)
        prv_file_path = str('{}/temp_cleaned.prv'.format(self.root_dir))  # Cleaned prv path
        in_place = True

        if not needed_info['status']:
            line_beginning_character.extend('1')
        if not needed_info['communication']:
            line_beginning_character.extend('3')
        if not needed_info['computation']:
            line_beginning_character.extend('2')

        print("INFO: The lines which are started with {}, have been eliminated.".format(line_beginning_character))

        # Delete the point_to_point communication (50000001)
        # Delete not needed Computation
        for computation_item in not_needed_computation:
            point_to_point_cmd = str(
                "awk -F: '!((NF==" + '8) &&' +
                ' ($7 =="{}"))'.format(computation_item) +
                "' {} > temp_cleaned.prv && mv temp_cleaned.prv {}/temp_cleaned.prv".format(
                    prv_file_path,
                    self.root_dir))
            print('The lines with the information about {} are eliminated.'.format(computation_item))
            subprocess.check_output(point_to_point_cmd, shell=True)

        # Delete not related lines (etc. comments,header,...)
        if in_place:
            eliminate_cmd = str(
                'sed -i "{};" {}'.format(';'.join('/^{}/d'.format(*t) for t in line_beginning_character),
                                         prv_file_path))
        else:
            eliminate_cmd = str(
                'sed "{};" {} > {}/temp_cleaned.prv'.format(
                    ';'.join('/^{}/d'.format(*t) for t in line_beginning_character),
                    prv_file_path,
                    self.root_dir))
        subprocess.check_output(eliminate_cmd, shell=True)
        return self

    def __get_events_index_list(self):
        """Get the .PRV events indicators

        Returns:
            A set of the events indicators <cpu:thread>.
        """
        point_to_point_cmd = str(
            "awk -F: '{print $2." + '"."' + '$3 "." $4 "." $5}' +
            "' {}/temp_cleaned.prv".format(self.root_dir))

        index_list = subprocess.check_output(point_to_point_cmd, shell=True).decode("utf-8")
        self.events_index_list = list(set(index_list.splitlines()))

        # Pickling a copy to multi proses
        with open('{}/events_index_list.pkl'.format(self.root_dir), 'wb') as f:
            pickle.dump(self.events_index_list, f)

        return self.events_index_list

    def shatter(self):
        """ It is shattered the .prv/text file to broadcast among the processes/threads.

        Returns:
            Self, it is applicable as a chain
        """
        # Todo: run the Paraver filter
        # Fixme: the last line of any chunk has to be repeated to the next one.

        # Clean previous outputs
        if os.path.isdir(self.root_dir):
            warnings.warn('Warning: The temporal directory already exist. It will be deleted and recreated by parser.')
            shutil.rmtree(self.root_dir)
        if os.path.isfile(self.csvpath):
            os.remove(self.csvpath)

        # Recreate the directory
        os.makedirs(self.root_dir)
        prv_file_path = str('{}/temp_cleaned.prv'.format(self.root_dir))  # Cleaned prv path
        self.__prv_cleaner()
        self.__get_events_index_list()

        # Compute the number of lines
        line_numbers_command = "wc -l < " + prv_file_path
        lines_number = int(subprocess.check_output(line_numbers_command, shell=True))
        self.__chunks = self._multi_process * self._multi_threads
        self.__chunk_size = round(lines_number / self.__chunks)

        if self.stop is None:
            self.stop = self.__chunk_size
        else:
            pass

        print("The parser is run over {} processes , {} lines per process and stop after parsing {} lines.".format(
            self.__chunks, self.__chunk_size, self.stop))

        # Create chunked in many .prv files
        split_command = 'split -l {} --numeric-suffixes --additional-suffix=.prv {} {}'.format(self.__chunk_size,
                                                                                               prv_file_path,
                                                                                               self.filepath)
        subprocess.check_output(split_command, shell=True)
        return self

    def _parse(self):
        """
        It tunes multi processes and runs the parsers in any of them.

        Functions:
            __drop_zero_rows(data, feature_set, feature_subset): Drop the rows with the zero values.
            __update_output_file(): Integrate the data in to a .csv file.
            __chunk_pars(chunk_name, parser_object): Set up the arguments and run the parser objects

        Returns:
            self, it is applicable as a chain
        """

        def __drop_zero_rows(data, feature_set=[], feature_subset=['PAPI_TOT_CYC', 'PAPI_TOT_INS']):
            """
            Drop the rows with the zero values

            Args:
                data (pandas): The raw dataset.
                feature_set (list): A list of the feature index
                feature_subset (list): a list of the feature name

            Returns:
                 A pandas dataframe
            """
            feature_subset_index = [feature_set.index(i) for i in feature_subset]
            return data[~(data[feature_subset_index] == 0).all(axis=1)]

        def __update_output_file():
            """
            Integrate the data into a .csv file.

            Returns:
                A string indicates the .csv file path
            """
            # Obtain the header
            with open('{}/csvh.pkl'.format(self.root_dir), 'rb') as pkl_header:
                self.csv_header = pickle.load(pkl_header)

            # Write the output as CSV dataset
            with open(self.csvpath, mode='w') as outputCSV:
                output_csv_writer = csv.writer(outputCSV)
                output_csv_writer.writerow(self.csv_header)  # Insert the header in to the .csv file.
                for csv_file in self.__csv_chunks:
                    chunk_data_final = pd.read_csv(str(self.root_dir + '/' + csv_file + '.csv'),
                                                   index_col=None,
                                                   header=None)
                    # delete zero rows (By default: 'PAPI_TOT_CYC'=0, 'PAPI_TOT_INS'=0)
                    chunk_data_final = __drop_zero_rows(data=chunk_data_final,
                                                        feature_set=self.csv_header,
                                                        feature_subset=['PAPI_TOT_CYC', 'PAPI_TOT_INS'])

                    # Write to the CSV
                    output_csv_writer.writerows(chunk_data_final.values)

            # Test the final csv shape.
            chunk_data_final = pd.read_csv(self.csvpath, index_col=None, header=None)
            print('Info: the final csv shape is {}.'.format(chunk_data_final.shape))

            # TODO: Delete the temporal files
            if os.path.isdir(self.root_dir):
                warnings.warn('The /temp/ directory will have been not needed. It has been deleted.')
                shutil.rmtree(self.root_dir)

            return self.csvpath

        def __chunk_pars(chunk_name, parser_object):
            """
            Set up the arguments and run the parser objects

            Args:
                chunk_name (string): The associated prv file to the chunk
                parser_object (prvparse): An associated prvparse object to the chunk

            Returns:
                 self, it is applicable as a chain
            """
            sys.argv[2] = str(self.root_dir) + '/' + str(chunk_name)
            parser_object(stop=self.stop)
            return self

        # Distribution configuration.
        if not self.__isDistributed:  # < 1 thread, 1 processes >
            warnings.warn('The parsing is not distributed, It will executed by one possessor/thread!!!', UserWarning)
            args = prvparse.Parser()
            args(stop=self.stop)
            # Test the final csv shape.
            chunk_data = pd.read_csv(self._output_path, index_col=None, header=None)
            print('The chunk data shape is {}.'.format(chunk_data.shape))

        elif self._multi_process > 1:
            chunk_list = filter(lambda x: x.startswith('fragment'), os.listdir(self.root_dir))
            if self._multi_threads == 1:  # < 1 thread, Multi processes >
                parser_list = [prvparse.Parser(chunk_id=chunk) for chunk in range(self._multi_process)]
                self.__csv_chunks = [re.sub(r'\W', '', str(csv_chunk).split()[-1]) for csv_chunk in parser_list]

                processes = [multiprocessing.Process(target=__chunk_pars, args=(chunk, parser_object)) for
                             chunk, parser_object in zip(chunk_list, parser_list)]

                for p in processes:  # Run processes
                    p.start()

                for p in processes:  # Exit the completed processes
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
        return self


def distributing():
    # DevelopTime: In order to reduce the draft time, Distributor(stop=3496) should be used.
    try:
        distributed = Distributor()
        distributed()
        print(BColors.OKGREEN + "The trace file {} is parsed successfully to a .csv file.".format(distributed._path) +
              BColors.ENDC)
    except ValueError as error:
        print(BColors.FAIL + "ERROR: The parsing proses is failed. {}".format(error) + BColors.ENDC)


if __name__ == '__main__':
    distributing()
