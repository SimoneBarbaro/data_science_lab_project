import os
import pandas as pd

from src.data.read_data import get_old_spider_data, dirname


def clean_spider_data():
    data = get_old_spider_data()
    data = data.drop(columns=["scores_here252"])
    data.to_csv(os.path.join(dirname, "../../data/spider_twosides_table.csv"), index=False, header=True)


def merge_spider():
    data1 = pd.read_excel("data/spider_twosides_table.xlsx").set_index(["alldrugs_TWOSIDES"])
    data2 = pd.read_excel("data/alldrugs_missing_TWOSIDES.xlsx").set_index(["alldrugs_TWOSIDES"])
    data = data1.combine_first(data2)
    data.reset_index().to_excel("data/spider_twosides_table.xlsx")


if __name__ == "__main__":
    merge_spider()
    clean_spider_data()
