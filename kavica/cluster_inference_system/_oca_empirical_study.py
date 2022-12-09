#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Organization Component Analysis Empirical Studies

It is a Cluster Self Organization Map Inference Systems

This module is an Artificial Neural base Cluster Inference System. It infers the cluster structure by the leverages
of the Self Organization Map.

Examples:
    $ python3 cluster_organization.py clustering_data.csv --cw feature_x feature_y -c cluster_number
                                                        --s map_size_x map_size_y -i epoch_number
        cluster_data.csv: Includes the clustering data
        --cw (str): Indicates the two features that have been used in order to cluster data (feature_x, feature_y)
        -c (str): Indicates the cluster that we would like to apply the organization analysis on.
        --s (int): Indicates the Self Organization Map grid sizes (map_size_x, map_size_y)
        -i (int): Indicates the ANN's training iterations.

Notes:
    It is applicable over both the original or the clustered data.


See:
    - Mahdavi, K., Mancho, J. L., & Lucas, J. G. (2021, July). Organization Component Analysis: The method for
      extracting insights from the shape of cluster. In 2021 International Joint Conference on Neural Networks
      (IJCNN) (pp. 1-10). IEEE.

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 07/11/2022
"""

from OCA import organization_component_analysis


def oca_experiment():
    """ Applies the OCA (Organization Component Analysis) of clusters

    Returns:
    """

    # Gromacs the first 5 clusters command
    # python3 _oca_empirical_study.py config/OCA_configs/gromacs.json ../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv --cw 'L1_Rat' 'n_PAPI_TOT_INS' -c 6 --s 10 10 -i 500
    # python3 _oca_empirical_study.py config/OCA_configs/gromacs.json ../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv --cw 'L1_Rat' 'n_PAPI_TOT_INS' -c 9 --s 13 13 -i 400
    # python3 _oca_empirical_study.py config/OCA_configs/gromacs.json ../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv --cw 'L1_Rat' 'n_PAPI_TOT_INS' -c 10 --s 13 13 -i 400
    # python3 _oca_empirical_study.py config/OCA_configs/gromacs.json ../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv --cw 'L1_Rat' 'n_PAPI_TOT_INS' -c 7 --s 12 12 -i 400
    # python3 _oca_empirical_study.py config/OCA_configs/gromacs.json ../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv --cw 'L1_Rat' 'n_PAPI_TOT_INS' -c 8 --s 10 10 -i 500

    # python3 _oca_empirical_study.py config/OCA_configs/iris.json ../../data/oca_test_data/iris.csv --cw 'sepal_width' 'sepal_length' -c 1 --s 6 6 -i 700

    # Feature enginering: https://cseweb.ucsd.edu/classes/wi15/cse255-a/reports/wi15/Yerlan_Idelbayev.pdf
    # python3 OCA.py config/OCA_configs/covtype.json empirical_report/OCA_paper_dataset/covtype.csv --cw 'HDT_Hydro' 'Elevation' -c 1 --s 8 8 -i 1000
    # python3 OCA.py config/OCA_configs/covtype.json empirical_report/OCA_paper_dataset/covtype_labeled.csv --cw 'HDT_Hydro' 'Elevation' -c 0 --s 12 12 -i 300

    # python3 OCA.py config/OCA_configs/home.json empirical_report/OCA_paper_dataset/home.csv --cw 'Temperature' 'Humidity' -c 1  --s 12 12 -i 300

    # python3 OCA.py config/OCA_configs/home.json empirical_report/OCA_paper_dataset/home.csv --cw 'Temperature' 'Humidity' -c 1 --s 9 9 -i 8000

    # python3 OCA.py config/OCA_configs/wine.json empirical_report/OCA_paper_dataset/wine.csv --cw 'OD280/OD315' 'Proline' -c 1 --s 8 8 -i 3000

    # https://archive.ics.uci.edu/ml/datasets/Letter+Recognitio
    # python3 OCA.py config/OCA_configs/letter-recognition.json empirical_report/OCA_paper_dataset/letter-recognition.csv --cw 'X-box' 'Y-box' -c 1 --s 8 8 -i 3000

    # https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/real-world/segment.arff
    # http://vis-www.cs.umass.edu/  (the data discription)
    # python3 OCA.py config/OCA_configs/segment.json empirical_report/OCA_paper_dataset/segment.csv --cw 'exgreen-mean' 'rawgreen-mean' -c 5 --s 12 12 -i 3000

    # python3 OCA.py config/OCA_configs/earthquakes_labeled.json empirical_report/OCA_paper_dataset/earthquakes_labeled.csv --cw 'location.longitude' 'location.latitude' -c 0 --s 10 10 -i 1000

    organization_component_analysis()


if __name__ == '__main__':
    oca_experiment()
