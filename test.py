#!/usr/bin/env python

from tqdm import tqdm

import flowrisk as fr


class Config(fr.BulkConfVPINConfig):

    N_TIME_BAR_FOR_INITIALIZATION = 50

    N_BUCKET_OR_BUCKET_DECAY = 0.95
    BUCKETS = fr.bulk.RecursiveBulkClassEWMABuckets


if __name__ == '__main__':
    config = Config()
    config.summary()

    example = fr.examples.USStocks(config)

    symbols = example.list_symbols('small')
    for symbol in tqdm(symbols):
        result = example.estimate_vpin_and_conf_interval(symbol)
        example.draw_price_vpins_and_conf_intervals()

    symbols = example.list_symbols('large')
    for symbol in tqdm(symbols):
        result = example.estimate_vpin_and_conf_interval(symbol)
        example.draw_price_vpins_and_conf_intervals()
