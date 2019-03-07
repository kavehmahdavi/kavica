
# -*- coding: utf-8 -*-
"""
Kavica is HPC data mining package that includes:
    - Trace (.txt) file parser
    - Missing value imputation
    - Factor analysis
    - Transformation
    - Cluster analysis
    -
    - Fourier analysis
    - Utility (used by subpackages)
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 17/01/2019

------------------------------------------------------------------------------------------------------------------------
Preparation:
------------------------------------------------------------------------------------------------------------------------
    - Install python3
    - Copy the package folder to your destination directory.
    - Install required packages with: pip install -r requirements.txt --user
    - In the root directory, run: $ pip3 install -e . --user (install the kavica package from the local source)
    - Activate Virtual environment: source /apps/PYTHON/3.6.1_virt/bin/activate
    - Set memory:
        - Incrise the stak size: $ ulimit -s 32000
        - Asking for nodes : $ If u
------------------------------------------------------------------------------------------------------------------------
Parsing the trace file (.prv):
------------------------------------------------------------------------------------------------------------------------
    - Compile the Cython code:  $ python3 kavica/parser/build.py
    - Create a config file (.json). E.g:
            {
              "clustered": boolean, 
              "thread_numbers": int,
              "hardware_counters": 
              {
                "IPC": "IPC",
                "42000050": "PAPI_TOT_INS",
                "42000059": "PAPI_TOT_CYC",
                ...
              },
              "call_stacks": 
              {
                "70000001": "Caller_at_level_1",
                "70000002": "Caller_at_level_2",
                "70000003": "Caller_at_level_3",
                "80000001": "Caller_line_at_level_1",
                "80000002": "Caller_line_at_level_2",
                "80000003": "Caller_line_at_level_3"
              }
            }
    - Parsing: $ python3 prv2csv.py config.json trace.prv -o output.csv -mp 2
        It scatter the .prv file among -mp (e.g 2) possesses and it produces output.csv and source.hdf5 as output.
        Example: $ python3 prv2csv.py config/config1.json ../../data/gromacs_64p.L12-CYC.prv -o output.csv -mp 10
------------------------------------------------------------------------------------------------------------------------
Interpolation: (missing value imputation)
------------------------------------------------------------------------------------------------------------------------
    - Create a config file (.json). E.g:
        {
          "scale": false,
          "hardware_counters": {
            "IPC": "IPC",
            "42000050": "PAPI_TOT_INS",
            "42000059": "PAPI_TOT_CYC",
            "42000053": "PAPI_LD_INS",..
          },
          "complimentary": {
            "70000001": "Caller_at_level_1",",
            "80000001": "Caller_line_at_level_1", ...
          },
          "pass_through": {
            "Object_id": "Object_id",
            "Timestamp": "Timestamp",
            "Active_hardware_counter_set": "Active_hardware_counter_set",
            "90000001": "label"
          }
        }
    - Impute the missing values: $  python3 mice.py config.json source2.csv -m 'norm' -o output.csv -i 10
        Example:  python3 mice.py config.json source2.csv -m 'norm' -o ../../data/output.csv -i 12
------------------------------------------------------------------------------------------------------------------------       
Feature Selection (Feature Analysis Methods):
------------------------------------------------------------------------------------------------------------------------
    - Create a config file (.json). E.g:
            {
              "thread_numbers": 64,
              "missing_values": "mean",
              "scale": true,
              "hardware_counters": 
              {
                "IPC": "IPC",
                "42000050": "PAPI_TOT_INS",
                "42000059": "PAPI_TOT_CYC",
                "42000053": "PAPI_LD_INS",
                "42000054": "PAPI_SR_INS",
                "42000006": "PAPI_L1_TCM",
                "42000007": "PAPI_L2_TCM",
                "42000008": "PAPI_L3_TCM",
                "42000021": "PAPI_TLB_IM"
              }
            }
    - In order to select the feature subset use: $ python3 feature_analysis.py config.json data.csv -k 2 -m PFA/IFA
        Example: python3 feature_analysis.py config/config.json ../parser/source.csv -k 3 -m IFA
------------------------------------------------------------------------------------------------------------------------
Feature Selection (Spectral Method):
------------------------------------------------------------------------------------------------------------------------
    - Create a config file (.json). E.g:
            {
              "thread_numbers": 64,
              "missing_values": "mean",
              "scale": true,
              "hardware_counters": 
              {
                "IPC": "IPC",
                "42000050": "PAPI_TOT_INS",
                "42000059": "PAPI_TOT_CYC",
                "42000053": "PAPI_LD_INS",
                "42000054": "PAPI_SR_INS",
                "42000006": "PAPI_L1_TCM",
                "42000007": "PAPI_L2_TCM",
                "42000008": "PAPI_L3_TCM",
                "42000021": "PAPI_TLB_IM"
              }
            }
    - In order to select the feature subset use: $ python3 spectral_methods.py config.json data.csv -k 2 -m LS/MCFS/SPEC
        Example: - $ python3 spectral_methods.py config/config.json ../parser/source1.csv -k 2 -m spec -bsize 6000
------------------------------------------------------------------------------------------------------------------------
