import os

from src.data.read_data import get_spider_data, get_old_tiger_data, get_tiger_data, dirname  #, FILTERED_SPIDER_DATA, FILTERED_TIGER_DATA, PROCESSED_SPIDER_DATA, PROCESSED_TIGER_DATA
from data.utils import ORIGINAL_TIGER_DATA


def clean_spider_data():
    data = get_spider_data()
    # data.to_csv(os.path.join(dirname, "../../data/{}".format(PROCESSED_SPIDER_DATA)), index=False, header=True)


def clean_tiger_data():
    data = get_old_tiger_data()
    data.to_csv(os.path.join(dirname, "../../data/{}".format(ORIGINAL_TIGER_DATA)), index=False, header=True)
    # data.to_csv(os.path.join(dirname, "../../data/{}".format(PROCESSED_TIGER_DATA)), index=False, header=True)


def clean_spider_tiger_data():
    spider = get_spider_data()
    tiger = get_tiger_data()
    spider[spider.index.isin(tiger.index)].to_csv(os.path.join(dirname, "../../data", FILTERED_SPIDER_DATA), index=True,
                                                  header=True)
    tiger[tiger.index.isin(spider.index)].to_csv(os.path.join(dirname, "../../data", FILTERED_TIGER_DATA), index=True,
                                                 header=True)


if __name__ == "__main__":
    clean_spider_data()
    clean_tiger_data()
    clean_spider_tiger_data()
