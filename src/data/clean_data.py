import os

from src.data.read_data import get_old_spider_data, dirname, PROCESSED_DATA


def clean_spider_data():
    data = get_old_spider_data()
    data.to_csv(os.path.join(dirname, "../../data/{}".format(PROCESSED_DATA)), index=False, header=True)

"""
Deprecated
def merge_spider():
    data1 = pd.read_excel(os.path.join(dirname, "../../data/spider_twosides_table.xlsx")).drop(columns="scores_here252").set_index(["mol_id", "alldrugs_TWOSIDES"])
    data2 = pd.read_excel(os.path.join(dirname, "../../data/alldrugs_missing_TWOSIDES.xlsx")).set_index(["mol_id", "alldrugs_TWOSIDES"])
    data = data1.combine_first(data2)
    data.reset_index().to_csv(os.path.join(dirname, "../../data/spider_twosides_data.csv"))
"""

if __name__ == "__main__":
    # merge_spider()
    clean_spider_data()
