#!/usr/bin/env python
#
# Created by Xixuan on Nov 22, 2018
#

import os
import pandas as pd

from flowrisk.toxicity.vpin import BulkConfVPINConfig, BulkConfVPIN


class DataLoader(object):

    FILE_MAPPER = dict()

    def __init__(self):
        pass

    def list_symbols(self):
        """
        Return the symbols.
        :rtype:     list
        """
        return self.FILE_MAPPER.keys()

    def load_data(self, symbol):
        """
        Load the data for the symbol and output to a dataframe.
        :param str symbol:      Symbol of the stock
        :rtype:                 pandas.DataFrame
        """
        assert symbol in self.FILE_MAPPER.keys(), \
            'symbol should be one of %s' % ', '.join(self.FILE_MAPPER.keys())
        data_path = self.FILE_MAPPER[symbol]
        return pd.read_csv(data_path, sep=',')


class USSmallCapsDataLoader(DataLoader):

    FILE_MAPPER = {
        'CBIO': os.path.join(os.path.dirname(__file__), 'data', 'CBIO_2018_11_22.csv'),
        'FBNC': os.path.join(os.path.dirname(__file__), 'data', 'FBNC_2018_11_22.csv'),
        'GNC': os.path.join(os.path.dirname(__file__), 'data', 'GNC_2018_11_22.csv'),
        'NDLS': os.path.join(os.path.dirname(__file__), 'data', 'NDLS_2018_11_22.csv'),
        'QES': os.path.join(os.path.dirname(__file__), 'data', 'QES_2018_11_22.csv'),
    }


class USLargeCapsDataLoader(DataLoader):

    FILE_MAPPER = {
        'V': os.path.join(os.path.dirname(__file__), 'data', 'V_2018_11_22.csv'),
        'AAPL': os.path.join(os.path.dirname(__file__), 'data', 'AAPL_2018_11_22.csv'),
        'NVDA': os.path.join(os.path.dirname(__file__), 'data', 'NVDA_2018_11_22.csv'),
        'GS': os.path.join(os.path.dirname(__file__), 'data', 'GS_2018_11_22.csv'),
        'INTC': os.path.join(os.path.dirname(__file__), 'data', 'INTC_2018_11_22.csv'),
    }


class USStocks(object):
    def __init__(self, config):
        """
        Instantiate an object of showing VPIN and its confidence intervals.
        :param BulkConfVPINConfig config:       Configuration for BulkConfVPIN
        """
        self.config = config

        self.config.TIME_BAR_TIME_STAMP_COL_NAME = 'date'
        self.config.TIME_BAR_PRICE_COL_NAME = '4. close'
        self.config.TIME_BAR_VOLUME_COL_NAME = '5. volume'

        self.vpin_estimator = BulkConfVPIN(config)

        self.us_small_cap_data_loader = USSmallCapsDataLoader()
        self.us_large_cap_data_loader = USLargeCapsDataLoader()

        self.all_available_symbols = (
                self.us_large_cap_data_loader.list_symbols()
                +
                self.us_small_cap_data_loader.list_symbols()
        )

        self.data = None
        self.vpins_and_conf_intervals = None

    def list_symbols(self, cap):
        """
        List the symbols of available data for 'large' or 'small' cap stocks.
        :param str cap:     'large' or 'small'
        :rtype:             list
        """
        assert cap in ('large', 'small'), \
            'cap should be either large or small'

        if cap == 'large':
            return self.us_large_cap_data_loader.list_symbols()
        else:
            return self.us_small_cap_data_loader.list_symbols()

    def estimate_vpin_and_conf_interval(self, symbol, draw=False):
        """
        Estimate VPIN values for a stock and calculate the associated confidence intervals.
        :param str symbol:      Symbol of the stock for VPINs
        :param bool draw:       Whether to make a plot for the VPINs and confidence intervals
        :rtype:                 pandas.DataFrame
        """
        assert symbol in self.all_available_symbols, \
            'symbol should be one of %s' % ', '.join(self.all_available_symbols)

        if symbol in self.us_large_cap_data_loader.list_symbols():
            self.data = self.us_large_cap_data_loader.load_data(symbol)
        else:
            self.data = self.us_small_cap_data_loader.load_data(symbol)

        self.config.SYMBOL = symbol
        self.vpins_and_conf_intervals = self.vpin_estimator.estimate(self.data)

        if draw:
            self.vpin_estimator.plot()

        return self.vpins_and_conf_intervals

    def draw_price_vpins_and_conf_intervals(self, out_to_file=True):
        """
        Put prices and vpins together in a single plot.
        :rtype:     matplotlib.axes._subplots.AxesSubplot
        """
        assert self.data is not None and self.vpins_and_conf_intervals is not None, \
            'has not estimated VPINs and confidence intervals for any symbols'

        ax1 = self.vpin_estimator.plot()
        ax1.locator_params(axis='x', nbins=10)

        ax2 = ax1.figure.add_subplot(111, sharex=ax1, frameon=False)

        scaled_volumes = self.data.loc[:, ['date', self.config.TIME_BAR_VOLUME_COL_NAME]]
        scaled_volumes.loc[:, self.config.TIME_BAR_VOLUME_COL_NAME] -= (
            scaled_volumes.loc[:, self.config.TIME_BAR_VOLUME_COL_NAME].min()
        )
        scaled_volumes.loc[:, self.config.TIME_BAR_VOLUME_COL_NAME] /= (
                scaled_volumes.loc[:, self.config.TIME_BAR_VOLUME_COL_NAME].max() + 1e-8
        )
        price_max = self.data.loc[:, self.config.TIME_BAR_PRICE_COL_NAME].max()
        price_min = self.data.loc[:, self.config.TIME_BAR_PRICE_COL_NAME].min()
        scaled_volumes.loc[:, self.config.TIME_BAR_VOLUME_COL_NAME] *= (price_max - price_min) / 3.0
        scaled_volumes.loc[:, self.config.TIME_BAR_VOLUME_COL_NAME] += price_min

        scaled_volumes.plot(
            x='date',
            y=self.config.TIME_BAR_VOLUME_COL_NAME,
            kind='bar',
            color='gray',
            ax=ax2,
            ylim=(price_min, price_max)
        )
        ax2.locator_params(axis='x', nbins=10)

        self.data.plot(
            x='date',
            y=self.config.TIME_BAR_PRICE_COL_NAME,
            ax=ax2,
            style='k-.',
            ylim=(price_min, price_max)
        )
        ax2.figure.autofmt_xdate()
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel("US $")

        if out_to_file:
            if not os.path.exists('./pics'):
                os.makedirs('./pics')
            ax1.figure.savefig('./pics/%s.png' % self.config.SYMBOL)

        return ax1
