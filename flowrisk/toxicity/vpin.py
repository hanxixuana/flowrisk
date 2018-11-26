#!/usr/bin/env python
#
#   Created by Xixuan on Nov 20, 2018
#

import numpy as np
import pandas as pd

from flowrisk.measure import RecursiveMeasure, BulkMeasure
from flowrisk.tools.vol import Vol, RecursiveEWMAVol
from flowrisk.toxicity.bulk import Buckets, RecursiveBulkClassMABuckets
from flowrisk.tools.band import Band, RecursiveEWMABand


class Config(object):

    SYMBOL = ''

    def __init__(self):
        pass

    def summary(self):
        print("\nConfigurations:\n")
        for attr in dir(self):
            if not attr.startswith("__"):
                if not callable(getattr(self, attr)):
                    print(
                        "{:30} {}".format(attr, getattr(self, attr))
                    )
                else:
                    if attr.upper() == attr:
                        print(
                            "{:30} {}".format(attr, getattr(self, attr).__name__)
                        )
        print("\n")


class RecursiveVPINConfig(Config):

    VOL_DECAY = 0.8
    VOL_ESTIMATOR = RecursiveEWMAVol

    N_BUCKET_OR_BUCKET_DECAY = 20
    BUCKET_MAX_VOLUME = 100000.0
    BUCKETS = RecursiveBulkClassMABuckets

    TIME_BAR_TIME_STAMP_COL_NAME = 'time'
    TIME_BAR_PRICE_COL_NAME = 'price'
    TIME_BAR_VOLUME_COL_NAME = 'volume'

    def check_params(self):
        assert issubclass(self.BUCKETS, Buckets), \
            'BUCKETS should be an inherited class from Buckets'
        assert issubclass(self.VOL_ESTIMATOR, Vol), \
            'VOL_ESTIMATOR should be an inherited class from Vol'
        return True


class RecursiveVPIN(RecursiveMeasure):
    def __init__(self, config):
        """
        Instantiate an object of RecursiveVPIN that recursively estimates VPIN.
        :param RecursiveVPINConfig config:       Configuration of the recursive estimator
        """
        config.check_params()
        self.config = config

        self.buckets = self.config.BUCKETS(
            self.config.BUCKET_MAX_VOLUME,
            self.config.N_BUCKET_OR_BUCKET_DECAY,
        )
        self.vol_estimator = self.config.VOL_ESTIMATOR(
            self.config.VOL_DECAY, 'pnl'
        )
        self.latest_time = None
        self.latest_vpin = None

    def initialize_vol_estimator(self, *args, **kwargs):
        self.vol_estimator.initialize(*args, **kwargs)

    def initialize_buckets(self, *args, **kwargs):
        self.buckets.initialize(*args, **kwargs)

    def check_time_bar(self, time_bar):
        assert isinstance(time_bar, pd.Series), \
            'time_bar should be a pandas.Series'
        assert self.config.TIME_BAR_TIME_STAMP_COL_NAME in time_bar.index, \
            'time_bar does not have a time stamp'
        assert self.config.TIME_BAR_PRICE_COL_NAME in time_bar.index, \
            'time_bar does not have a price'
        assert self.config.TIME_BAR_VOLUME_COL_NAME in time_bar.index, \
            'time_bar does not have a volume'

    def update(self, time_bar):
        """
        Update the measure using a new time bar.
        :param pd.Series time_bar:      New time bar of the market information
        :rtype:                         float
        """
        self.check_time_bar(time_bar)

        vol_estimate = self.vol_estimator.update(
            time_bar[self.config.TIME_BAR_PRICE_COL_NAME]
        )
        self.buckets.update(
            time_bar[self.config.TIME_BAR_PRICE_COL_NAME],
            vol_estimate,
            time_bar[self.config.TIME_BAR_VOLUME_COL_NAME]
        )

        self.latest_time = time_bar[self.config.TIME_BAR_TIME_STAMP_COL_NAME]
        self.latest_vpin = (
                np.sum(
                    self.buckets.get_order_imbalance()
                )
                /
                np.sum(
                    self.buckets.get_bucket_volume()
                )
        )

        return self.latest_vpin


class RecursiveConfVPINConfig(RecursiveVPINConfig):

    VPIN_MEAN_EWMA_DECAY = 0.9
    VPIN_VOL_EWMA_DECAY = 0.99
    VPIN_CONF_INTERVAL_RADIUS = 2.0
    BAND_ESTIMATOR = RecursiveEWMABand

    def check_params(self):
        assert issubclass(self.BUCKETS, Buckets), \
            'BUCKETS should be an inherited class from Buckets'
        assert issubclass(self.VOL_ESTIMATOR, Vol), \
            'VOL_ESTIMATOR should be an inherited class from Vol'
        assert issubclass(self.BAND_ESTIMATOR, Band), \
            'BAND_ESTIMATOR should be an inherited class from Band'
        return True


class RecursiveConfVPIN(RecursiveVPIN):
    def __init__(self, config):
        """
        Instantiate an object of RecursiveVPIN that recursively estimates VPIN.
        :param RecursiveConfVPINConfig config:       Configuration of the recursive estimator
        """
        super(RecursiveConfVPIN, self).__init__(config)

        self.config = config
        self.band_estimator = self.config.BAND_ESTIMATOR(
            self.config.VPIN_MEAN_EWMA_DECAY,
            self.config.VPIN_VOL_EWMA_DECAY,
            self.config.VPIN_CONF_INTERVAL_RADIUS
        )

    def update(self, time_bar):
        """
        Update the measure using a new time bar.
        :param pd.Series time_bar:      New time bar of the market information
        :rtype:                         float
        """
        self.check_time_bar(time_bar)

        vol_estimate = self.vol_estimator.update(
            time_bar[self.config.TIME_BAR_PRICE_COL_NAME]
        )
        self.buckets.update(
            time_bar[self.config.TIME_BAR_PRICE_COL_NAME],
            vol_estimate,
            time_bar[self.config.TIME_BAR_VOLUME_COL_NAME]
        )

        self.latest_time = time_bar[self.config.TIME_BAR_TIME_STAMP_COL_NAME]
        self.latest_vpin = float(
                np.sum(
                    self.buckets.get_order_imbalance()
                )
                /
                np.sum(
                    self.buckets.get_bucket_volume()
                )
        )

        if not self.band_estimator.is_initialized():
            self.band_estimator.initialize(self.latest_vpin, 0.0)

        confidence_interval = self.band_estimator.update(self.latest_vpin)

        return self.latest_vpin, confidence_interval


class BulkVPINConfig(RecursiveConfVPINConfig):

    N_TIME_BAR_FOR_INITIALIZATION = 2


class BulkVPIN(BulkMeasure):

    def __init__(self, config):
        """
        Instantiate an object of BulkVPIN that estimates a series of VPIN estimates.
        :param BulkVPINConfig config:       Configuration of the BulkVPIN estimator
        """
        config.check_params()
        self.config = config

        self.vpins = None
        self.recursive_vpin_estimator = RecursiveVPIN(self.config)

    @staticmethod
    def check_data(data):
        assert isinstance(data, pd.DataFrame), \
            'data should be a pandas.DataFrame'
        assert len(data) > 2, \
            'the length of data should be larger than 1'

    def estimate(self, data):
        """
        Estimate the measure using time series data.
        :param pd.DataFrame data:       DataFrame of market data
        :rtype:                         pd.DataFrame
        """
        self.check_data(data)

        self.recursive_vpin_estimator.initialize_vol_estimator(
            data.ix[
                data.index[:self.config.N_TIME_BAR_FOR_INITIALIZATION],
                self.config.TIME_BAR_PRICE_COL_NAME
            ].values
        )
        self.recursive_vpin_estimator.initialize_buckets(
            data.ix[
                data.index[:self.config.N_TIME_BAR_FOR_INITIALIZATION],
                self.config.TIME_BAR_PRICE_COL_NAME
            ].values,
            self.recursive_vpin_estimator.vol_estimator.get_latest_vol(),
            data.ix[
                data.index[:self.config.N_TIME_BAR_FOR_INITIALIZATION],
                self.config.TIME_BAR_VOLUME_COL_NAME
            ].values,
        )

        vpin_list = None
        first_idx = data.index[self.config.N_TIME_BAR_FOR_INITIALIZATION]
        for idx, row in data.loc[self.config.N_TIME_BAR_FOR_INITIALIZATION:].iterrows():
            vpin_value = self.recursive_vpin_estimator.update(row)
            if idx == first_idx:
                vpin_list = [vpin_value for _ in range(self.config.N_TIME_BAR_FOR_INITIALIZATION + 1)]
            else:
                vpin_list.append(vpin_value)

        self.vpins = data[self.config.TIME_BAR_TIME_STAMP_COL_NAME].to_frame()
        self.vpins['vpin'] = vpin_list
        return self.vpins

    def plot(self):
        """
        Make a plot for the time series of VPIN values.
        :rtype:     matplotlib.axes._subplots.AxesSubplot
        """
        assert self.vpins is not None, \
            'Has not estimated VPIN values yet'
        ax = self.vpins.plot(
            x='date',
            y='vpin',
            title=(
                    '%sVPIN from %s to %s' %
                    (
                        self.config.SYMBOL + ' ' if len(self.config.SYMBOL) else self.config.SYMBOL,
                        self.vpins['date'].values[0],
                        self.vpins['date'].values[-1]
                    )
            ),
            figsize=[12, 6.75]
        )
        ax.figure.autofmt_xdate()
        ax.set_ylabel("Percentage")
        return ax


class BulkConfVPINConfig(BulkVPINConfig):
    pass


class BulkConfVPIN(BulkVPIN):
    def __init__(self, config):
        """
        Instantiate an object of BulkConfVPIN that estimates a series of VPIN estimates
        and its confidence intervals
        :param BulkConfVPINConfig config:       Configuration of the BulkConfVPIN estimator
        """
        super(BulkConfVPIN, self).__init__(config)

        self.config = config
        self.recursive_vpin_estimator = RecursiveConfVPIN(self.config)
        self.confidence_intervals = None

    def estimate(self, data):
        """
        Estimate the measure using time series data.
        :param pd.DataFrame data:       DataFrame of market data
        :rtype:                         pd.DataFrame
        """
        self.check_data(data)

        self.recursive_vpin_estimator.initialize_vol_estimator(
            data.ix[
                data.index[:self.config.N_TIME_BAR_FOR_INITIALIZATION],
                self.config.TIME_BAR_PRICE_COL_NAME
            ].values
        )
        self.recursive_vpin_estimator.initialize_buckets(
            data.ix[
                data.index[:self.config.N_TIME_BAR_FOR_INITIALIZATION],
                self.config.TIME_BAR_PRICE_COL_NAME
            ].values,
            self.recursive_vpin_estimator.vol_estimator.get_latest_vol(),
            data.ix[
                data.index[:self.config.N_TIME_BAR_FOR_INITIALIZATION],
                self.config.TIME_BAR_VOLUME_COL_NAME
            ].values,
        )

        vpin_list = None

        lower_band = None
        mean = None
        upper_band = None

        first_idx = data.index[self.config.N_TIME_BAR_FOR_INITIALIZATION]

        for idx, row in data.loc[self.config.N_TIME_BAR_FOR_INITIALIZATION:].iterrows():
            vpin_value, confidence_interval_with_mean = self.recursive_vpin_estimator.update(row)
            if idx == first_idx:
                vpin_list = [vpin_value for _ in range(self.config.N_TIME_BAR_FOR_INITIALIZATION + 1)]
                lower_band = [
                    confidence_interval_with_mean[0]
                    for _ in range(self.config.N_TIME_BAR_FOR_INITIALIZATION + 1)
                ]
                mean = [
                    confidence_interval_with_mean[1]
                    for _ in range(self.config.N_TIME_BAR_FOR_INITIALIZATION + 1)
                ]
                upper_band = [
                    confidence_interval_with_mean[2]
                    for _ in range(self.config.N_TIME_BAR_FOR_INITIALIZATION + 1)
                ]
            else:
                vpin_list.append(vpin_value)
                lower_band.append(confidence_interval_with_mean[0])
                mean.append(confidence_interval_with_mean[1])
                upper_band.append(confidence_interval_with_mean[2])

        self.vpins = data[self.config.TIME_BAR_TIME_STAMP_COL_NAME].to_frame()
        self.vpins['vpin'] = vpin_list
        self.vpins['vpin_lower_band'] = lower_band
        self.vpins['vpin_mean'] = mean
        self.vpins['vpin_upper_band'] = upper_band
        return self.vpins

    def plot(self):
        """
        Make a plot for the time series of VPIN values.
        :rtype:     matplotlib.axes._subplots.AxesSubplot
        """
        assert self.vpins is not None, \
            'Has not estimated VPIN values yet'
        ax = self.vpins.plot(
            x='date',
            y='vpin',
            style='b',
            title=(
                    '%sVPIN from %s to %s' %
                    (
                        self.config.SYMBOL + ' ' if len(self.config.SYMBOL) else self.config.SYMBOL,
                        self.vpins['date'].values[0],
                        self.vpins['date'].values[-1]
                    )
            ),
            figsize=[12, 6.75]
        )
        self.vpins.plot(
            x='date',
            y='vpin_mean',
            ax=ax,
            style='r',
            lw=2.0
        )
        self.vpins.plot(
            x='date',
            y='vpin_lower_band',
            ax=ax,
            style='g--',
        )
        self.vpins.plot(
            x='date',
            y='vpin_upper_band',
            ax=ax,
            style='g--',
        )
        ax.figure.autofmt_xdate()
        ax.set_ylabel("Percentage")
        return ax
