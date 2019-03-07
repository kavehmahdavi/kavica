import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

# read the configuration file for preparing the features
def __configoration(config, data):
    # read the configuration file
    with open(config, 'r') as config:
        config_dict = json.load(config)

    # Read the data file
    df = pd.read_csv(data)

    columns_order = list(df.columns.values)
    active_features = list(set(list(config_dict['hardware_counters'].values())
                               + list(config_dict['complimentary'].values())))

    pass_through_features = list(set(list(config_dict['pass_through'].values())
                                     + list(config_dict['complimentary'].values())))

    # config the data set based on configuration information
    df = df[active_features]  # sub set of features
    return df, pass_through_features, columns_order

# read the configuration file for preparing the features
def __configoration_FS(config, data):
    # read the configuration file
    with open(config, 'r') as config:
        config_dict = json.load(config)

    # Read the data file
    df = pd.read_csv(data)

    # config the data set based on configuration information
    df = df[list(config_dict['hardware_counters'].values())]  # sub set of features
    df = df.replace([np.inf, -np.inf], np.nan)
    lastShape = df.shape

    # Remove the all zero rows
    df = df[(df.T != 0).any()]
    print("The {} row are full null that are eliminated.".format(lastShape[0] - df.shape[0]))
    lastShape = df.shape

    # Remove all NaN columns.
    df = df.ix[:, (pd.notnull(df)).any()]
    print("The {} columns are full null that are eliminated.".format(lastShape[1] - df.shape[1]))

    if config_dict['missing_values'] == 'mean':
        df.fillna(df.mean(), inplace=True)

    if config_dict['scale']:
        df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
    print(df.mean(axis=0), df.std(axis=0))

    # fixme: it is just reset the indexing for draft
    df = df.reset_index()

    return df


# -----------------------------------------------------------
# ---------------------Import Kavica-------------------------
# -----------------------------------------------------------
from kavica.transformer import BatchTransformer,VerticalTransformer  # It needs some minor tuning
from kavica.feature_selector import feature_analysis, spectral_methods
from kavica.imputation import mice
from kavica.parser import prv2csv


# -----------------------------------------------------------
# ------------------parse a trace file/s---------------------
# -----------------------------------------------------------
arguments = [
    'kavica/parser/config/config_gromacs_64p_L12-CYC.json',
    'data/gromacs_64p.L12-CYC.prv',
    '-o',
    'output.csv',
    '-mp',
    '100'
    ]
sys.argv.extend(arguments)
distributed = prv2csv.Distributor()
distributed()
print("\033[32mThe trace file {} is parsed successfully to both .hdf5 and .csv file.".format(distributed._path))


input()
# -----------------------------------------------------------
# ---------------------interpolation-------------------------
# -----------------------------------------------------------
mice_config='kavica/imputation/config/config_IM_gromacs_64p.json'
csv_file='source.csv'
output_path='kavica/parser/output.csv'
df, features_appending_list, columns_order = __configoration(mice_config, csv_file)

# Manipulating the data -> add synthetic missing value in df
df['PAPI_TOT_INS'][2040]= np.NaN
df['PAPI_LD_INS'][2040]= np.NaN
df['IPC'][2040]= np.NaN

df_clean = mice.Mice(df, predictMethod='norm.nob', iteration=90)
df_clean()
df_clean._write_csv(output_path=output_path,
               appendTo=features_appending_list,
               csvPath=csv_file,
               order=columns_order)


# -----------------------------------------------------------
# ------------------Load cleaned data frame------------------
# -----------------------------------------------------------
feature_selection_config='kavica/feature_selector/config/config_FS_gromacs_64p_INS_CYC.json'
df = __configoration_FS(feature_selection_config,output_path)

input()
# -----------------------------------------------------------
# ------------feature selection (factor analysis)------------
# -----------------------------------------------------------

# ---------------------------PFA-----------------------------
featureSelectionModel = feature_analysis.PrincipalFeatureAnalysis(k=3)
featureSelectionModel._rank_features(df.drop(axis=1, labels=['index','Duration']), dendrogram=True)
print(featureSelectionModel._feature_score_table().table)
print("\033[32mThe feature selection process is successfully completed by {} method.".format(
    featureSelectionModel.featureScore.get("method")))


# --------------------------IFA------------------------------
featureSelectionModel = feature_analysis.IndependentFeatureAnalysis(k=3)
featureSelectionModel._rank_features(df.drop(axis=1, labels=['index','Duration']), dendrogram=True)
print(featureSelectionModel._feature_score_table().table)
print("\033[32mThe feature selection process is successfully completed by {} method.".format(
    featureSelectionModel.featureScore.get("method")))

input()
# -----------------------------------------------------------
# ------------feature selection (spectral graph)-------------
# -----------------------------------------------------------

# ---------------------------LS----------------------------
featureSelectionModel = spectral_methods.LaplacianScore()
featureSelectionModel.fit(df, bag_size=2000)
featureSelectionModel.rank_features()
print("\n", featureSelectionModel.featureScoure)
print(featureSelectionModel.feature_score_table().table)
print("\033[32mThe feature selection process is successfully completed by {} method.".format(
    featureSelectionModel.featureScoure.get("method")))


# ---------------------------MCFS----------------------------
featureSelectionModel = spectral_methods.MultiClusterScore(k=2)
featureSelectionModel.fit(df, bag_size=2000)
featureSelectionModel.rank_features()
print("\n", featureSelectionModel.featureScoure)
print(featureSelectionModel.feature_score_table().table)
print("\033[32mThe feature selection process is successfully completed by {} method.".format(
    featureSelectionModel.featureScoure.get("method")))


# ---------------------------SPEC----------------------------
featureSelectionModel = spectral_methods.SPEC(k=2)
featureSelectionModel.fit(df, bag_size=3400)
featureSelectionModel.rank_features()
print("\n", featureSelectionModel.featureScoure)
print(featureSelectionModel.feature_score_table().table)
print("\033[32mThe feature selection process is successfully completed by {} method.".format(
featureSelectionModel.featureScoure.get("method")))