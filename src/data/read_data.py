import os
import pandas as pd

dirname = os.path.dirname(__file__)  # Trying to fix the path problems

FILTERED_INFIX = "_filtered"
NAMES_SUFFIX = "_names"

ORIGINAL_SPIDER_DATA = "alldrugs_twosides_revised_spider.csv"  # old: "alldrugs_twosides_merged.csv", "spider_twosides_table.xlsx"
# PROCESSED_SPIDER_DATA = "alldrugs_twosides_table.csv"  # old: "spider_twosides_table.csv"
FILTERED_SPIDER_DATA = "spider_filtered_with_tiger.csv"
SPIDER_MATRIX_SAMPLE = "matrix_spider{}_{}_{}{}.pkl.gz"
SPIDER_MATRIX_FULL = "matrix_spider{}_full{}.pkl.gz"

ORIGINAL_TIGER_DATA = "tiger_twosides_data.csv"
FILTERED_TIGER_DATA = "tiger_filtered_with_spider.csv"
# PROCESSED_TIGER_DATA = "tiger_twosides_table.csv"
TIGER_MATRIX_SAMPLE = "matrix_tiger{}_{}_{}{}.pkl.gz"
TIGER_MATRIX_FULL = "matrix_tiger{}_full{}.pkl.gz"


def get_old_tiger_data():
    """
    This function is deprecated, don't use it for further processing.
    Load the tiger dataset as it is.
    :return: A pandas dataframe containing the spider dataset.
    """
    data = pd.read_excel(os.path.join(dirname, "../../data", "alldrugs_TWOSIDES__MW150-750_TIGER.xlsm"))\
        .rename(columns={"# COMPOUND_ID": "alldrugs_TWOSIDES"})
    return data


def get_spider_data(filtered=False):
    if filtered:
        data = pd.read_csv(os.path.join(dirname, "../../data", FILTERED_SPIDER_DATA),
                           index_col="alldrugs_TWOSIDES")
    else:
        data = pd.read_csv(os.path.join(dirname, "../../data", ORIGINAL_SPIDER_DATA),
                           index_col="alldrugs_TWOSIDES").drop(columns=["mol_id"])
    return data


def get_tiger_data(filtered=False):
    if filtered:
        data = pd.read_csv(os.path.join(dirname, "../../data", FILTERED_TIGER_DATA),
                           index_col="alldrugs_TWOSIDES")
    else:
        data = pd.read_csv(os.path.join(dirname, "../../data", ORIGINAL_TIGER_DATA),
                           index_col="alldrugs_TWOSIDES")
    return data

def create_matrix(data):
    """
    Create the matrix of the pair interactions from a dataset of interactions.
    :param data: A dataset of interactions.
    :return: A new dataframe where each distinct pair of rows is summed together along the columns.
    """
    X = []
    I = []
    for i1, (n1, d1) in enumerate(data.iterrows()):
        for i2, (n2, d2) in enumerate(data.iterrows()):
            if i1 < i2:
                I.append({'name1': n1, 'name2': n2})
                X.append(d1 + d2)
    return pd.DataFrame(X).reset_index(drop=True), pd.DataFrame(I).reset_index(drop=True)


def load_sample_with_names(dataset, frac=1, random_state=1, filtered=False, save=False):
    """
    Load a fraction of the data with corresponding drug pair names.
    :param dataset: string specifying whether to use the 'spider' or 'tiger' dataset
    :param frac: fraction of data from the original Excel file
    :param random_state: seed
    :param filtered: whether to consider only drug pairs shared between SPiDER and TIGER datasets
    :param save: whether to save (and subsequently load) the output dataframes as compressed pickles (specified by frac and random_state) in the data folder
    :return:
        - data_matrix: dataframe matrix of interactions (from create_matrix)
        - matrix_names: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix
    """
    if frac == 1:
        return load_full_matrix_with_names(dataset)

    if dataset == "spider":
        path = os.path.join(dirname, "../../data", SPIDER_MATRIX_SAMPLE.format(FILTERED_INFIX if filtered else "",
                                                                               frac, random_state, ""))
        names_path = os.path.join(dirname, "../../data", SPIDER_MATRIX_SAMPLE.format(FILTERED_INFIX if filtered else "",
                                                                                     frac, random_state, NAMES_SUFFIX))
    elif dataset == "tiger":
        path = os.path.join(dirname, "../../data", TIGER_MATRIX_SAMPLE.format(FILTERED_INFIX if filtered else "",
                                                                              frac, random_state, ""))
        names_path = os.path.join(dirname, "../../data", TIGER_MATRIX_SAMPLE.format(FILTERED_INFIX if filtered else "",
                                                                                    frac, random_state, NAMES_SUFFIX))
    else:
        raise AttributeError("dataset {} not found".format(dataset))

    if save and os.path.exists(path) and os.path.exists(names_path):
        data_matrix = pd.read_pickle(path)
        matrix_names = pd.read_pickle(names_path)
    else:
        get_data_fn = get_spider_data if dataset == "spider" else get_tiger_data
        data_sample_name = get_data_fn(filtered).sample(frac=frac, random_state=random_state)
        data_matrix, matrix_names = create_matrix(data_sample_name)
        if save:
            data_matrix.to_pickle(path)
            matrix_names.to_pickle(names_path)
    return data_matrix, matrix_names


def load_full_matrix_with_names(dataset, filtered=False):
    """
    Load the full data matrix with corresponding drug pair names.
    :param dataset: string specifying whether to use the 'spider' or 'tiger' dataset
    :param filtered: whether to consider only drug pairs shared between SPiDER and TIGER datasets
    :return:
        - data_full: dataframe matrix of interactions (from create_matrix)
        - names_full: two-column dataframe with the names of drug pairs corresponding to the rows of the matrix
    """
    if dataset == "spider":
        path = os.path.join(dirname, "../../data", SPIDER_MATRIX_FULL.format(FILTERED_INFIX if filtered else "", ""))
        names_path = os.path.join(dirname, "../../data", SPIDER_MATRIX_FULL.format(FILTERED_INFIX if filtered else "", NAMES_SUFFIX))
    elif dataset == "tiger":
        path = os.path.join(dirname, "../../data", TIGER_MATRIX_FULL.format(FILTERED_INFIX if filtered else "", ""))
        names_path = os.path.join(dirname, "../../data", TIGER_MATRIX_FULL.format(FILTERED_INFIX if filtered else "", NAMES_SUFFIX))
    else:
        raise AttributeError("dataset {} not found".format(dataset))
    if os.path.exists(path) and os.path.exists(names_path):
        data_full = pd.read_pickle(path)
        names_full = pd.read_pickle(names_path)
    else:
        get_data_fn = get_spider_data if dataset == "spider" else get_tiger_data
        data_simple = get_data_fn(filtered)
        data_full, names_full = create_matrix(data_simple)
        data_full.to_pickle(path)
        names_full.to_pickle(names_path)
    return data_full, names_full

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
