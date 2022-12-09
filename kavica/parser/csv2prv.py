#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Base class to convert CSV to PRV (Converter & Manipulator)

A PRV file is generated base on a CSV file or an existing PRV file is manipulated based on a CSV file data.

The csv file most have these columns:
    "Object_id": The process number e.g. 9.1.9.1
    "Timestamp": The timestamp of the event
    "Hardware_Counter": The HWC code to identify which HWC should be changed
    "Values": The associated value to the HWC
    "Tag": Indicates the action that you are interested to be done
            - a: Append a HWC:value at the end of a line
            - u: Update the value of an existing HWC in a line
            - n: New line added to the file. The line includes a just a HWC:Value

Example:
    $ python csv2prv.py ../../data/parser_test_data/manipulating_list.csv -m ../../data/parser_test_data/gromacs_64p.chop1.prv -o manipulated.prv -mp 32

Dependency: # Todo: apply after each change in the __init__.py, path, and directories
    $ cd kavica_container
    $ python3 setup.py install

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 28/09/2022
"""

import argparse
import multiprocessing
import os.path
import subprocess
import sys
import warnings
import pandas as pd
import pickle
import shutil
from kavica.utils import BColors

__all__ = ['Distributor']


# Fixme: pandas ix replace with loc/iloc
class Distributor(object):
    """
    Shrink the data file and distribute it among the processes for parsing.
    """

    def __init__(self, stop=None):
        self._multi_threads = 1
        self._multi_process = 1
        self.__chunks = None
        self.__chunk_size = None
        self._input_csv_path = None
        self._input_prv_path = None
        self._output_prv_path = os.path.join('../../data/parser_test_data/manipulated.prv')
        self.__isDistributed = False
        self.stop = stop
        self.root_dir = '../temp'  # Unique chunk directory
        self.filepath = '../temp/fragment'
        self.__csv_chunks = []
        self.csv_header = []
        self.events_index_list = []
        self.split_indexes = None
        self.manipulating_df = None
        self.prv_lines = None
        self.line_pointer = None

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
                '../../data/parser_test_data/manipulating_list.csv',
                '-m',
                '../../data/parser_test_data/gromacs_64p.chop1.prv',
                # '../../data/scaling/HO.48.4its.prv',
                '-o',
                'manipulated.prv',
                '-mp',
                '32'
            ]
            sys.argv.extend(arguments)

        # parse the arguments
        parser = argparse.ArgumentParser(description='The files that are needed for manipulating.')
        parser.add_argument('csvfile', help='A .csv file use either for manipulating or reconstructing a .PRV file.')
        parser.add_argument('-m',
                            dest='m',
                            action='store',
                            type=str,
                            help="It needs a .PRV trace file (path) in order to manipulate.")
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

        self._input_csv_path = args.csvfile
        if args.o:
            self._output_prv_path = args.o
        if args.m:
            self._input_prv_path = args.m
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

    # TODO: it is not needed for manipulating but it will be needed by reconstructor.
    def __get_events_index_list(self):
        """Get the .PRV events indicators

        Returns:
            A set of the events indicators <cpu,thread>.
        """
        point_to_point_cmd = str("awk -F: '{print $2." + '"."' + '$3 "." $4 "." $5}' +
                                 "' {}/temp_cleaned.prv".format(self.root_dir))

        index_list = subprocess.check_output(point_to_point_cmd, shell=True).decode("utf-8")
        self.events_index_list = list(set(index_list.splitlines()))

        # Pickling to multi proses
        with open('{}/events_index_list.pkl'.format(self.root_dir), 'wb') as f:
            pickle.dump(self.events_index_list, f)

        return self.events_index_list

    def get_split_indexes(self, df, column_name='Timestamp'):
        """Compute the pandas df split indexes.

        It computes the indexes and the Timestamp of a row to split a pandas df.

        Args:
            df (pandas): Include the input data
            column_name (str): Indicates the name of the column of df that is used for splitting.

        Returns:
            A dict when the keys are the row index and the values are timestamp.
        """
        chunk_index = [0]
        chunk_timestamp = [df.ix[0, [column_name]][column_name]]
        for chunk_item in range(1, self._multi_process):
            item_index = int(df.shape[0] / self._multi_process) * chunk_item
            item_timestamp = df.ix[item_index, [column_name]][column_name]
            while item_timestamp == chunk_timestamp[-1]:
                item_index = item_index + 1
                item_timestamp = df.ix[item_index, [column_name]][column_name]
            chunk_index.append(item_index)
            chunk_timestamp.append(item_timestamp)

        return dict(zip(chunk_index[1:], chunk_timestamp[1:]))

    def shatter(self):
        """ It is shattered the manipulating .csv and .prv file among the processes/threads.

        Returns:
            Self, it is applicable as a chain
        """
        # Todo: run the Paraver filter
        # Todo: the last line of any chunk has to be repeated to the next one.

        # Clean previous results
        if os.path.isdir(self.root_dir):
            warnings.warn('The temporal directory already exist. It will be deleted and recreated by parser.')
            shutil.rmtree(self.root_dir)
        if os.path.isfile(self._output_prv_path):
            os.remove(self._output_prv_path)

        # Create new directory
        os.makedirs(self.root_dir)

        # find the line index for splitting the file based on the manipulating action
        self.manipulating_df = pd.read_csv(self._input_csv_path)
        self.manipulating_df['Tag'] = self.manipulating_df['Tag'].str.lower()
        self.split_indexes = self.get_split_indexes(self.manipulating_df)

        # Read the number of lines
        line_numbers_command = "wc -l < " + self._input_prv_path
        self.prv_lines = int(subprocess.check_output(line_numbers_command, shell=True))

        # split the main file
        self.line_pointer = ['0']
        for chunk_item in self.split_indexes.values():
            # find the line pointer
            command_line = "awk -F: '{if($1=='2' && $6=='" + \
                           "{}') ".format(chunk_item) + \
                           "{print NR} 1}' " + \
                           "{}".format(self._input_prv_path)
            chunk_upper_bound = max(
                subprocess.check_output(command_line, shell=True).decode("ascii").strip().split('\n'))

            # Compute the chunk file
            command_line = "awk 'NR >= {} && NR < {}' {} > {}/manipulate_chunk{}to{}.txt".format(self.line_pointer[-1],
                                                                                                 chunk_upper_bound,
                                                                                                 self._input_prv_path,
                                                                                                 self.root_dir,
                                                                                                 self.line_pointer[-1],
                                                                                                 chunk_upper_bound)
            subprocess.check_output(command_line, shell=True)

            # Update the line_pointer list
            self.line_pointer.append(chunk_upper_bound)

        command_line = "awk 'NR >= {}' {} > {}/manipulate_chunk{}to{}.txt".format(self.line_pointer[-1],
                                                                                  self._input_prv_path,
                                                                                  self.root_dir,
                                                                                  self.line_pointer[-1],
                                                                                  self.prv_lines)
        subprocess.check_output(command_line, shell=True)
        return self

    def manipulate(self):
        """ It tunes multi processes and runs the parser over of them.

        Functions:
            __chunk_manipulator(chunk_name, lower_bound, upper_bound): Apply the prv manipulation
        """

        def __chunk_manipulator(chunk_name, lower_bound, upper_bound):
            """ Set up the arguments and run the parser objects

            Args:
                chunk_name (str): The chunk file name
                lower_bound (int): The starting row index
                upper_bound (int): The finishing row index

            Returns:
                A boolean which is True when the manipulation was finished successfully

            """
            _tag_dict = {'a': 'Add', 'u': 'Update', 'n': 'NewLine'}
            partial_manipulating_list = self.manipulating_df.iloc[lower_bound:upper_bound + 1]
            for row_index, manipulating_item in partial_manipulating_list.iterrows():
                task_id = manipulating_item['Object_id'].split('.')

                # Update HWC of the existing row.
                if manipulating_item['Tag'] in ['u', 'U']:
                    # It works to a line that has the HWC and we want to replace it.
                    command_line = "gawk -i inplace -F: '($1=='2' && $2=='{}' && $3=='{}' && $4=='{}' && $5=='{}' && $6=='{}') ".format(
                        task_id[0],
                        task_id[1],
                        task_id[2],
                        task_id[3],
                        manipulating_item['Timestamp']) \
                                   + '{' + 'sub(/{}:[0-9]*/,"{}:{}")'.format(manipulating_item['Hardware_Counter'],
                                                                             manipulating_item['Hardware_Counter'],
                                                                             manipulating_item['Values']) \
                                   + "} 1'" + " {}/{}".format(self.root_dir, chunk_name)

                # Add the hardware counter:value at the end of a line
                # fixme: it adds the key:value at the end of each line but a line should be the target
                elif manipulating_item['Tag'] in ['a', 'A']:
                    command_line = "gawk -i inplace -F: '($1=='2' && $2=='{}' && $3=='{}' && $4=='{}' && $5=='{}' && $6=='{}') ".format(
                        task_id[0],
                        task_id[1],
                        task_id[2],
                        task_id[3],
                        manipulating_item['Timestamp']) \
                                   + '{' + '$0=$0":{}:{}"'.format(manipulating_item['Hardware_Counter'],
                                                                  manipulating_item['Values']) \
                                   + "}1'" + " {}/{}".format(self.root_dir, chunk_name)

                # Add a line with a single counter:value
                elif manipulating_item['Tag'] in ['n', 'N']:
                    command_line_find = "gawk -F: '$6~/{}/".format(manipulating_item['Timestamp']) \
                                        + "{print NR,$1,$2,$3,$4,$5,$6}'" \
                                        + " {}/{}".format(self.root_dir, chunk_name)
                    # ('line', 'id_0', 'id_1', 'id_2', 'id_3', 'id_4', 'timestamp')
                    _found_line = subprocess.check_output(command_line_find, shell=True).decode("utf-8").split()
                    command_line = "awk -i inplace -F: '($1=='{}' && $2=='{}' && $3=='{}' && $4=='{}' && $5=='{}' && " \
                                   "$6=='{}') ".format(_found_line[1],
                                                       _found_line[2],
                                                       _found_line[3],
                                                       _found_line[4],
                                                       _found_line[5],
                                                       manipulating_item['Timestamp']) \
                                   + '{' + '$0=$0"\\n{}:{}:{}:{}:{}:{}:{}:{}"'.format(2,
                                                                                      _found_line[2],
                                                                                      _found_line[3],
                                                                                      _found_line[4],
                                                                                      _found_line[5],
                                                                                      manipulating_item['Timestamp'],
                                                                                      manipulating_item[
                                                                                          'Hardware_Counter'],
                                                                                      manipulating_item['Values']) \
                                   + "} 1'" + " {}/{}".format(self.root_dir, chunk_name)
                else:
                    raise ValueError("Error: Not defined tag '{}', most be in [u,a,n]".format(manipulating_item['Tag']))

                subprocess.check_output(command_line, shell=True)
                print(BColors.OKBLUE + 'Item:<{},{}>'.format(row_index, _tag_dict.get(manipulating_item['Tag'])) \
                      + BColors.ENDC, command_line)

            print("Manipulation is finished over chunk {} in [{} to {}]".format(chunk_name, lower_bound, upper_bound))
            return True

        # Apply the change/replacement over the files
        lower_bounds = [0] + list(self.split_indexes.keys())
        upper_bounds = list(self.split_indexes.keys()) + [self.manipulating_df.shape[0]]
        file_prefix = self.line_pointer
        file_suffix = self.line_pointer[1:] + [self.prv_lines]

        chunk_list = []
        processes = []
        for lower_b, upper_b, suffix, prefix in zip(lower_bounds, upper_bounds, file_suffix, file_prefix):
            chunk_item = "manipulate_chunk{}to{}.txt".format(prefix, suffix)
            chunk_list.append(chunk_item)
            p = multiprocessing.Process(target=__chunk_manipulator,
                                        args=(chunk_item, lower_b, upper_b,))
            processes.append(p)
            p.start()
            # __chunk_manipulator(chunk_name=chunk_item, lower_bound=lower_b, upper_bound=upper_b)

        for process in processes:
            process.join()
            if p.is_alive():
                print("Job {} is not finished!".format(p))

        # Marge the files
        with open(self._output_prv_path, 'wb') as wfd:
            for f in chunk_list:
                with open('{}/{}'.format(self.root_dir, f), 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)

        # Remove CTRL-M (^M) characters from the file in Linux
        command_line__ctrl_m = 'sed -e "s/\r//g"' \
                               + " {} > temp_{}".format(self._output_prv_path, self._output_prv_path) \
                               + " ; mv temp_{} {}".format(self._output_prv_path, self._output_prv_path)
        subprocess.check_output(command_line__ctrl_m, shell=True)
        print(BColors.OKGREEN + "Remove CTRL-M (^M) characters from the file in Linux" + BColors.ENDC)

        # Todo: Multi thread or processes
        """
        # Handling the different distribution circumstances.
        if not self.__isDistributed:  # < 1 thread, 1 processes >
            warnings.warn('The parsing is not distributed, It will executed by one possessor/thread!!!', UserWarning)
            args = prvparse.Parser()
            args(stop=self.stop)
            # Test the final csv shape.
            chunk_data = pd.read_csv(self._output_path, index_col=None, header=None)
            print(chunk_data.shape)

        elif self._multi_process > 1:
            chunk_list = filter(lambda x: x.startswith('manipulate_chunk'), os.listdir(self.root_dir))

            if self._multi_threads == 1:  # < 1 thread, Multi processes >

                processes = [multiprocessing.Process(target=__chunk_manipulator(), args=(chunk, parser_object)) for
                             chunk in chunk_list]

                # Run processes
                for p in processes:
                    p.start()

                # Exit the completed processes
                for p in processes:
                    p.join()
                    if p.is_alive():
                        print("Job {} is not finished!".format(p))


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
        """

    # ---------------------------------------------------------------------------------------
    # Todo: Create a complete prv trace file from a csv.
    # Todo: It is not finished
    # ---------------------------------------------------------------------------------------
    def csv_converter(self):
        """ Convert a csv file to a prv trace file.

        Returns:

        """
        pass

    def add_row(self):
        print("ERROR: add_row function is a draft. Use manipulating function")
        exit()
        """ It tunes multi processes and runs the parser over of them."""
        print(BColors.WARNING +
              "You are in add row. To manipuliet an existing row change the flag in distributing(add_row=False)" +
              BColors.ENDC)

        def __chunk_manipulator(chunk_name, lower_bound, upper_bound):
            """ Set up the arguments and run the parser objects"""
            partial_manipulating_list = self.manipulating_df.iloc[lower_bound:upper_bound + 1]

            for row_index, manipulating_item in partial_manipulating_list.iterrows():
                task_id = manipulating_item['Object_id'].split('.')

                # Todo: it is working for a line that has the HWC and we want to rplace it.

                command_line = "awk -i inplace -F: '($1=='2' && $2=='{}' && $3=='{}' && $4=='{}' && $5=='{}' && $6=='{}') ".format(
                    task_id[0],
                    task_id[1],
                    task_id[2],
                    task_id[3],
                    manipulating_item['Timestamp']) \
                               + '{' + 'sub(/{}:[0-9]*/,"{}:{}")'.format(manipulating_item['Hardware_Counter'],
                                                                         manipulating_item['Hardware_Counter'],
                                                                         manipulating_item['Values']) \
                               + "} 1'" + " {}/{}".format(self.root_dir, chunk_name)
                subprocess.check_output(command_line, shell=True)

                if manipulating_item['Object_id'] == "0.0.0.0":
                    command_line_find = "gawk -F: '$6~/{}/".format(manipulating_item['Timestamp']) \
                                        + "{print NR,$1,$2,$3,$4,$5,$6}'" \
                                        + " {}/{}".format(self.root_dir, chunk_name)
                    # ('line', 'id_0', 'id_1', 'id_2', 'id_3', 'id_4', 'timestamp')
                    _finded_line = subprocess.check_output(command_line_find, shell=True).decode("utf-8").split()

                    # TODO: Insert the line befor the HW line
                    """
                    command_line_insert = "sed '{} i ".format(_finded_line[0]) \
                                          + '{}:{}:{}:{}:{}:{}:{}:{}'.format(_finded_line[1],
                                                                                    _finded_line[2],
                                                                                    _finded_line[3],
                                                                                    _finded_line[4],
                                                                                    _finded_line[5],
                                                                                    manipulating_item['Timestamp'],
                                                                                    manipulating_item[
                                                                                        'Hardware_Counter'],
                                                                                    manipulating_item['Values']) \
                                           + "'" + " {}/{}".format(self.root_dir, chunk_name)
                    print(command_line_insert)
                    subprocess.check_output(command_line_insert, shell=True)
                    """

                # Adding the hardware counter:value at the end of a line
                """
                command_line = "awk -i inplace -F: '($1=='{}' && $2=='{}' && $3=='{}' && $4=='{}' && $5=='{}' && " \
                               "$6=='{}') ".format(
                    _finded_line[1],
                    _finded_line[2],
                    _finded_line[3],
                    _finded_line[4],
                    _finded_line[5],
                    manipulating_item['Timestamp']) \
                               + '{' + '$0=$0"{}:{}:{}:{}:{}:{}:{}:{}"'.format(_finded_line[1],
                                                                               _finded_line[2],
                                                                               _finded_line[3],
                                                                               _finded_line[4],
                                                                               _finded_line[5],
                                                                               manipulating_item['Timestamp'],
                                                                               manipulating_item['Hardware_Counter'],
                                                                               manipulating_item['Values']) \
                               + "} 1'" + " {}/{}".format(self.root_dir, chunk_name)
                subprocess.check_output(command_line, shell=True)
                print(command_line)
                """
            print("Manipulation is finished over chunk {} in [{} to {}]".format(chunk_name,
                                                                                lower_bound,
                                                                                upper_bound))

        # Apply the change/replacement over the files
        lower_bounds = [0] + list(self.split_indexes.keys())
        upper_bounds = list(self.split_indexes.keys()) + [self.manipulating_df.shape[0]]
        file_prefix = self.line_pointer
        file_suffix = self.line_pointer[1:] + [self.prv_lines]

        chunk_list = []
        processes = []
        for lower_b, upper_b, suffix, prefix in zip(lower_bounds, upper_bounds, file_suffix, file_prefix):
            chunk_item = "manipulate_chunk{}to{}.txt".format(prefix, suffix)
            chunk_list.append(chunk_item)
            p = multiprocessing.Process(target=__chunk_manipulator,
                                        args=(chunk_item, lower_b, upper_b,))
            processes.append(p)
            p.start()
            # __chunk_manipulator(chunk_name=chunk_item, lower_bound=lower_b, upper_bound=upper_b)

        for process in processes:
            process.join()
            if p.is_alive():
                print("Job {} is not finished!".format(p))

        # Marge the files
        with open(self._output_prv_path, 'wb') as wfd:
            for f in chunk_list:
                with open('{}/{}'.format(self.root_dir, f), 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)

        # Todo: Multi thread or processes
        """
        # Handling the different distribution circumstances.
        if not self.__isDistributed:  # < 1 thread, 1 processes >
            warnings.warn('The parsing is not distributed, It will executed by one possessor/thread!!!', UserWarning)
            args = prvparse.Parser()
            args(stop=self.stop)
            # Test the final csv shape.
            chunk_data = pd.read_csv(self._output_path, index_col=None, header=None)
            print(chunk_data.shape)

        elif self._multi_process > 1:
            chunk_list = filter(lambda x: x.startswith('manipulate_chunk'), os.listdir(self.root_dir))

            if self._multi_threads == 1:  # < 1 thread, Multi processes >

                processes = [multiprocessing.Process(target=__chunk_manipulator(), args=(chunk, parser_object)) for
                             chunk in chunk_list]

                # Run processes
                for p in processes:
                    p.start()

                # Exit the completed processes
                for p in processes:
                    p.join()
                    if p.is_alive():
                        print("Job {} is not finished!".format(p))


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
        """


def distributing(add_row=False):
    distributed = Distributor()
    distributed.arguments_parser()
    distributed.shatter()
    if add_row:
        distributed.add_row()
    else:
        distributed.manipulate()

    print(BColors.OKGREEN + "The trace {} is manipulated successfully.".format(
        distributed._input_prv_path) + BColors.ENDC)


if __name__ == '__main__':
    distributing()

# Command line exmaple

# Gromacs
# python3 csv2prv.py ../../ECII_full_test/Gromacs/manipulating_list_tsen.csv -m ../../ECII_full_test/Gromacs/gromacs_64p.chop1.cl.IPCxINS.prv -o ../../ECII_full_test/Gromacs/gromacs_64p.chop1.cl.IPCxINS.FSCM.prv -mp 16

# Gromacs chop2
# python3 csv2prv.py ../../ECII_full_test/Gromacs/manipulating_list_FSCM_L1.csv -m ../../ECII_full_test/Gromacs/chope2/gromacs_64p.cl.0106m4.chop1.l1.prv -o ../../ECII_full_test/Gromacs/chope2/gromacs_64p.chop1.cl.L1.chop1.FSCM.prv -mp 16


# 4+4 lulesh
# python3 csv2prv.py row2prv_data/4kaveh_jesus/manipulating_list_original.csv -m row2prv_data/4kaveh_jesus/lulesh_4+4_with_uncores_sockets1+2.chop_5it.prv -o lulesh_4+4_with_uncores_sockets1+2.chop_5it_cluster.prv
