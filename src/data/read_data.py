import os
import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)  # Trying to fix the path problems
ORIGINAL_DATA = "alldrugs_twosides_merged.xlsx"  # old: "spider_twosides_table.xlsx"
PROCESSED_DATA = "alldrugs_twosides_merged.csv"  # old: "spider_twosides_table.csv"


def get_old_spider_data():
    """
    This function is deprecated, don't use it for further processing.
    Load the spider dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_csv(os.path.join(dirname, "../../data/{}".format(ORIGINAL_DATA)), index_col="alldrugs_TWOSIDES").drop(columns=["mol_id"])
    return data


def get_spider_data():
    """
    Load the spider dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_csv(os.path.join(dirname, "../../data/{}".format(PROCESSED_DATA)))
    return data


def get_spider_data_with_names():
    data = pd.read_csv(os.path.join(dirname, "../../data/{}".format(ORIGINAL_DATA)), index_col="alldrugs_TWOSIDES").drop(columns=["mol_id"])
    """
    data = pd.read_excel(os.path.join(dirname, "../../data/spider_twosides_table.xlsx"),
                         index_col="alldrugs_TWOSIDES").drop(columns=["mol_id", "scores_here252"])
    """
    return data


def create_matrix(data):
    """
    Create the matrix of the pair interactions from a dataset of interactions.
    :param data: A dataset of interactions.
    :return: A new dataframe where each distinct pair of rows is summed together along the columns.
    """
    X = []
    for i1, d1 in data.iterrows():
        for i2, d2 in data.iterrows():
            if i1 < i2:
                X.append(d1 + d2)
    return pd.DataFrame(X)


def load_sample(frac=1, random_state=1, save=False):
    if save and os.path.exists(os.path.join(dirname, "../../data/matrix_spider_{}_{}.npy".format(frac, random_state))):
        return np.load(os.path.join(dirname, "../../data/matrix_spider_{}_{}.npy".format(frac, random_state)))

    data_sample = get_spider_data_sample(frac=frac, random_state=random_state)
    data = create_matrix(data_sample).values
    if save:
        np.save(os.path.join(dirname, "../../data/matrix_spider_{}_{}.npy".format(frac, random_state)), data)
    return data


def load_sample_with_names(frac=1, random_state=1, save=False):
    """
    Load a fraction of the data with corresponding drug pair names.
    :param frac: fraction of data from the original Excel file
    :param random_state: seed
    :param save: whether to save (and subsequently load) the output dataframes as compressed pickles (specified by frac and random_state) in the data folder
    :return:
        - data_matrix: dataframe matrix of interactions (from create_matrix)
        - matrix_names: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix
    """
    if save and os.path.exists(os.path.join(dirname, "../../data/matrix_spider_{}_{}.pkl.gz".format(frac,
                                                                                                    random_state))) and os.path.exists(
            os.path.join(dirname, "../../data/matrix_spider_names_{}_{}.pkl.gz".format(frac, random_state))):
        data_matrix = pd.read_pickle(
            os.path.join(dirname, "../../data/matrix_spider_{}_{}.pkl.gz".format(frac, random_state)))
        matrix_names = pd.read_pickle(
            os.path.join(dirname, "../../data/matrix_spider_names_{}_{}.pkl.gz".format(frac, random_state)))
    else:
        data_sample_name = get_spider_data_with_names().sample(frac=frac, random_state=random_state)
        data_matrix = create_matrix(data_sample_name)
        matrix_names = get_drug_names(data_sample_name)
        if save:
            data_matrix.to_pickle(
                os.path.join(dirname, "../../data/matrix_spider_{}_{}.pkl.gz".format(frac, random_state)))
            matrix_names.to_pickle(
                os.path.join(dirname, "../../data/matrix_spider_names_{}_{}.pkl.gz".format(frac, random_state)))
    return data_matrix, matrix_names


def load_full_matrix_with_names():
    """
    Load the full data matrix with corresponding drug pair names.
    :return:
        - data_full: dataframe matrix of interactions (from create_matrix)
        - names_full: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix
    """
    if os.path.exists(os.path.join(dirname, "../../data/matrix_spider_full.pkl.gz")) and os.path.exists(
            os.path.join(dirname, "../../data/matrix_spider_names_full.pkl.gz")):
        data_full = pd.read_pickle(os.path.join(dirname, "../../data/matrix_spider_full.pkl.gz"))
        names_full = pd.read_pickle(os.path.join(dirname, "../../data/matrix_spider_names_full.pkl.gz"))
    else:
        data_simple = get_spider_data_with_names()
        data_full = create_matrix(data_simple)
        names_full = get_drug_names(data_simple)
        data_full.to_pickle(os.path.join(dirname, "../../data/matrix_spider_full.pkl.gz"))
        names_full.to_pickle(os.path.join(dirname, "../../data/matrix_spider_names_full.pkl.gz"))
    return data_full, names_full


def get_drug_names(data):
    """
    Create a pandas dataframe with names of drug pairs corresponding to create_matrix(data).
    :param data: A pandas dataframe of interactions, indexed with drug names.
    :return: A pandas dataframe of drug name pairs, name1 | name2.
    """
    names = data.index
    X = []
    for i1, n1 in pd.DataFrame(names).iterrows():
        for i2, n2 in pd.DataFrame(names).iterrows():
            if i1 < i2:
                X.append({'name1': n1[0], 'name2': n2[0]})
    return pd.DataFrame(X)


def get_drug_index():
    """
    Create a pandas dataframe with names and IDs of drug pairs corresponding to the matrix from create_matrix.
    :return: A pandas dataframe with ID of the first drug | name of the first drug | ID of the second drug | name of the second drug.
    """
    names = pd.read_csv(os.path.join(dirname, "../../data/{}".format(ORIGINAL_DATA))).iloc[:, 0:2]
    X = []
    for i1, d1 in names.iterrows():
        for i2, d2 in names.iterrows():
            if i1 < i2:
                X.append({'mol_id1': d1[0], 'name1': d1[1], 'mol_id2': d2[0], 'name2': d2[1]})
    return pd.DataFrame(X)


def get_spider_data_sample(**kwargs):
    """
    Load the spider dataset and take a random subset of it.
    :param ...: Parameters passed to sample. Recommended: frac and random_state.
    :return: A pandas dataframe containing a subset of the spider dataset.
    """
    data = get_spider_data().sample(**kwargs)
    return data


def get_twosides_meddra(pickle=True):
    """
    Load the the TWOSIDES database with medDRA descriptions.
    :param pickle: whether to read data/TWOSIDES_medDRA.pkl.gz (smaller, faster) instead of data/TWOSIDES_medDRA.csv.gz
    :return: the TWOSIDES database with SPiDER drug pairs and side effect classifications according to medDRA
    """
    if pickle:
        return pd.read_pickle(os.path.join(dirname, "../../data/TWOSIDES_medDRA.pkl.gz"))
    else:
        return pd.read_csv(os.path.join(dirname, "../../data/TWOSIDES_medDRA.csv.gz"))


def filter_twosides(data_matrix, data_names, twosides):
    """
    From data_matrix and data_names, filter the pairs that are present in the TWOSIDES database. 
    :param data_matrix: dataframe matrix of interactions (from create_matrix)
    :param data_names: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix
    :param twosides: the TWOSIDES database with columns drug_1_name and drug_2_name representing the drug pairs (as read by get_twosides_meddra)
    :return:
        - data_matrix_ts: dataframe matrix of only the interactions where the pair is present in TWOSIDES
        - data_names_ts: two-column dataframe of names of drug pairs present in TWOSIDES
    """
    twosides_names = twosides.filter(["drug_1_name", "drug_2_name"]).drop_duplicates()

    def concat_names(df, col1, col2):
        return pd.concat([df[col1] + df[col2], df[col2] + df[col1]])  # Columns in either order

    mask = (data_names["name1"] + data_names["name2"]).isin(concat_names(twosides_names, "drug_1_name", "drug_2_name"))

    return data_matrix.loc[mask], data_names.loc[mask]


def match_meddra(filtered_names, twosides):
    """
    Match the given drug pairs with their medDRA side effect descriptions.
    :param filtered_names: two-column dataframe of names of drug pairs present in TWOSIDES
    :param twosides: the TWOSIDES database with columns drug_1_name and drug_2_name representing the drug pairs (as read by get_twosides_meddra)
    Output

    :return: dataframe with the medDRA classifications of side effects for the drug pairs given in data_names_ts
    """

    def concat_names(df, col1, col2):
        return pd.concat([df[col1] + df[col2]])

    mask12 = (twosides["drug_1_name"] + twosides["drug_2_name"]).isin(concat_names(filtered_names, "name1", "name2"))
    mask21 = (twosides["drug_1_name"] + twosides["drug_2_name"]).isin(concat_names(filtered_names, "name2", "name1"))

    res = twosides.copy().loc[mask12 | mask21]
    # Swap pair order to match filtered_names
    res.loc[mask21, ['drug_1_name', 'drug_2_name']] = res.loc[mask21, ['drug_2_name', 'drug_1_name']].values
    res.rename(columns={"drug_1_name": "name1", "drug_2_name": "name2"}, inplace=True)

    if len(filtered_names.columns) > 2:
        res = res.merge(filtered_names, on=["name1", "name2"])

    return res
