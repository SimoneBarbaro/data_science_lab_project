import os

from src.data.read_data import get_old_spider_data, dirname


def clean_spider_data():
    data = get_old_spider_data()
    data = data.drop(columns=["scores_here252"])
    data.to_csv(os.path.join(dirname, "../../data/spider_twosides_table.csv"), index=False, header=True)


if __name__ == "__main__":
    clean_spider_data()
