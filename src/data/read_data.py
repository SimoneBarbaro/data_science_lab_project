import os
import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)  # Trying to fix the path problems


def get_old_spider_data():
    """
    This function is deprecated, don't use it for further processing.
    Load the spider dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_excel(os.path.join(dirname, "../../data/spider_twosides_table.xlsx")).set_index(["mol_id", "alldrugs_TWOSIDES"])
    return data


def get_spider_data():
    """
    Load the spider dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_csv(os.path.join(dirname, "../../data/spider_twosides_table.csv"))
    return data

def get_spider_data_with_names():
    data = pd.read_excel(os.path.join(dirname, "../../data/spider_twosides_table.xlsx"), index_col="alldrugs_TWOSIDES").drop(columns=["mol_id", "scores_here252"])
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
    if save and os.path.exists(os.path.join(dirname, f"../../data/matrix_spider_{frac}_{random_state}.npy")):
        return np.load(os.path.join(dirname, f"../../data/matrix_spider_{frac}_{random_state}.npy"))

    data_sample = get_spider_data_sample(frac=frac, random_state=random_state)
    data = create_matrix(data_sample).values
    if save:
        np.save(os.path.join(dirname, f"../../data/matrix_spider_{frac}_{random_state}.npy"), data)
    return data


def load_sample_with_names(frac=1, random_state=1, save=False):
    if save and os.path.exists(os.path.join(dirname, f"../../data/matrix_spider_{frac}_{random_state}.pkl")) and os.path.exists(os.path.join(dirname, f"../../data/matrix_spider_names_{frac}_{random_state}.pkl")):
        data_matrix = pd.read_pickle(os.path.join(dirname, f"../../data/matrix_spider_{frac}_{random_state}.pkl"))
        matrix_names = pd.read_pickle(os.path.join(dirname, f"../../data/matrix_spider_names_{frac}_{random_state}.pkl"))
    else:
        data_sample_name = get_spider_data_with_names().sample(frac=frac, random_state=random_state)
        data_matrix = create_matrix(data_sample_name)
        matrix_names = get_drug_names(data_sample_name)
        if save:
            data.to_pickle(os.path.join(dirname, f"../../data/matrix_spider_names_{frac}_{random_state}.pkl"), data)
    return data_matrix, matrix_names


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
    names = pd.read_excel(os.path.join(dirname, "../../data/spider_twosides_table.xlsx")).iloc[:, 0:2]
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