import os

from src.data.read_data import get_spider_data, get_old_tiger_data, dirname, \
    ORIGINAL_TIGER_DATA  # , PROCESSED_SPIDER_DATA, PROCESSED_TIGER_DATA


def clean_spider_data():
    data = get_spider_data()
    # data.to_csv(os.path.join(dirname, "../../data/{}".format(PROCESSED_SPIDER_DATA)), index=False, header=True)


def clean_tiger_data():
    data = get_old_tiger_data()
    data.to_csv(os.path.join(dirname, "../../data/{}".format(ORIGINAL_TIGER_DATA)), index=False, header=True)
    # data.to_csv(os.path.join(dirname, "../../data/{}".format(PROCESSED_TIGER_DATA)), index=False, header=True)


if __name__ == "__main__":
    clean_spider_data()
    clean_tiger_data()
