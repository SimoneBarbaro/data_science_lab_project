import pandas as pd


def get_old_spider_data():
    """
    This function is deprecated, don't use it for further processing.
    Load the spider dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_excel("../data/spider_twosides_table.xlsx").set_index(["mol_id", "alldrugs_TWOSIDES"])
    return data


def get_spider_data():
    """
    Load the spider dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_csv("../data/spider_twosides_table.csv")
    return data


def create_matrix(data):  # TODO
    """
    Create the matrix of the pair interactions from a dataset of interactions.
    :param data: A dataset of interactions.
    :return: A new dataframe where each distinct pair of rows is summed together along the columns.
    """
    X = []
    for d1 in data.iterrows():
        for d2 in data.iterrows():
            if d1[0] < d2[0]:
                X.append(d1[1] + d2[1])
    return pd.DataFrame(X)
