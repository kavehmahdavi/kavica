#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compiling and Building Cython of the .prv parse.
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 17/01/2019

import subprocess

annotation_command = "cython -a prvparse.pyx"
subprocess.call(annotation_command, shell=True)

# build the file
build_command = "python3 setup.py build_ext --inplace"
subprocess.call(build_command, shell=True)

# move the build file to the root
move_command_so = "mv kavica/parser/prvparse.cpython-36m-x86_64-linux-gnu.so ."
subprocess.call(move_command_so, shell=True)

move_command_o = "mv build/temp.linux-x86_64-3.6/prvparse.o ."
subprocess.call(move_command_o, shell=True)

# delete the temporal build folders
delete_command = "rm -r build/ kavica/"
subprocess.call(delete_command, shell=True)
