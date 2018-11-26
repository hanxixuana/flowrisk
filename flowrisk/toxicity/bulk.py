#!/usr/bin/env python
#
#   Created by Xixuan on Nov 20, 2018
#

import numpy as np

from scipy.stats import norm
from abc import ABCMeta, abstractmethod


class Buckets:

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_bucket_volume(self):
        """
        Return bucket_volume_array.
        :rtype:     numpy.ndarray or float
        """
        raise NotImplementedError

    @abstractmethod
    def get_order_imbalance(self):
        """
        Return order_imbalance.
        :rtype:     numpy.ndarray or float
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, price, pnl_volatility_estimate, volume):
        """
        Update a new time bar to a bucket in this series of buckets.
        :param float price:                         Price of a new time bar
        :param float pnl_volatility_estimate:       Current volatility estimate of the price pnl
        :param int or float volume:                 Volume of a new time bar
        """
        raise NotImplementedError


class RecursiveBulkClassMABuckets(Buckets):
    def __init__(self, bucket_max_volume, n_bucket):
        """
        Instantiate an object of Buckets used for volume bucketing and bulk classification in
        a MA style.
        :param int or float bucket_max_volume:      Maximum volume for each bucket (bucket size)
        :param int n_bucket:                        Maximum number of buckets for a single VPIN estimate
        """
        assert isinstance(bucket_max_volume, int) or isinstance(bucket_max_volume, float), \
            'bucket_max_volume should be an int'
        self.bucket_max_volume = bucket_max_volume
        assert isinstance(n_bucket, int), \
            'n_bucket should be an int'
        self.n_bucket = n_bucket

        self.current_bucket_idx = 0

        self.bucket_volume_array = np.zeros(self.n_bucket)
        self.bucket_buy_volume_array = np.zeros(self.n_bucket)
        self.bucket_sell_volume_array = np.zeros(self.n_bucket)

        self.latest_price = None

        self.current_bucket_volume_from_last_round = 0.0
        self.current_bucket_buy_volume_from_last_round = 0.0
        self.current_bucket_sell_volume_from_last_round = 0.0

    def is_initialized(self):
        """
        Check if the buckets is initialized..
        :rtype:     bool
        """
        return self.latest_price is not None

    @staticmethod
    def check_initial_args(initial_price_or_prices, pnl_volatility_estimate, initial_volumes):
        assert isinstance(initial_price_or_prices, float) or isinstance(initial_price_or_prices, np.ndarray), \
            'initial_price_or_prices should be a numpy.ndarray or a float'
        if isinstance(initial_price_or_prices, float):
            assert initial_price_or_prices >= 0.0, \
                'initial_price_or_prices should be non-negative if it is a float'
        assert isinstance(pnl_volatility_estimate, float) or pnl_volatility_estimate is None, \
            'pnl_volatility_estimate should be a float or None'
        if isinstance(pnl_volatility_estimate, float):
            assert pnl_volatility_estimate >= 0.0, \
                'pnl_volatility_estimate should be non-negative if it is a float'
        assert isinstance(initial_volumes, np.ndarray) or initial_volumes is None, \
            'initial_vol should be a numpy.ndarray or None'

    def initialize(self, initial_price_or_prices, pnl_volatility_estimate=None, initial_volumes=None):
        """
        Initialize the buckets in the bulk classification.
        1.  Provide an initial price
        2.  Provide a series of prices in a numpy.ndarray, a volatility estimate and a series of
            volumes in a numpy.ndarray.
        :param float or np.ndarray initial_price_or_prices:     Prices in time bars
        :param None or float pnl_volatility_estimate:           Current volatility estimate of the price pnl
        :param None or np.ndarray initial_volumes:              Volumes in time bars
        """
        self.check_initial_args(initial_price_or_prices, pnl_volatility_estimate, initial_volumes)
        if isinstance(initial_price_or_prices, np.ndarray):
            self.initialize_with_prices_and_volumes(
                initial_price_or_prices,
                pnl_volatility_estimate,
                initial_volumes
            )
        if isinstance(initial_price_or_prices, float):
            self.initialize_with_price(initial_price_or_prices)

    @staticmethod
    def check_prices_and_volumes(initial_prices, initial_volumes):
        assert len(initial_prices) >= 1, \
            'initial_prices should have more than one price'
        assert len(initial_prices.shape) == 1, \
            'initial_prices should be one-dimensional vector'
        assert initial_prices.dtype == 'float64' or initial_prices.dtype == 'float32', \
            'the datatype of initial_prices should be float32 or float64'
        assert len(initial_volumes) >= 1, \
            'initial_volumes should have more than one price'
        assert len(initial_volumes.shape) == 1, \
            'initial_volumes should be one-dimensional vector'
        assert initial_volumes.dtype == 'float64' or initial_volumes.dtype == 'float32', \
            'the datatype of initial_volumes should be float32 or float64'

    def initialize_with_prices_and_volumes(self, initial_prices, pnl_volatility_estimate, initial_volumes):
        """
        Initialize the buckets with an initial price for delta_p in the bulk classification.
        :param np.ndarray initial_prices:           Prices in time bars
        :param float pnl_volatility_estimate:       Current volatility estimate of the price pnl
        :param np.ndarray initial_volumes:          Volumes in time bars
        :return:
        """
        self.initialize_with_price(initial_prices.mean())
        for price, volume in zip(initial_prices, initial_volumes):
            self.update(price, pnl_volatility_estimate, volume)

    def initialize_with_price(self, init_price):
        """
        Initialize the buckets with an initial price for delta_p in the bulk classification.
        :param float init_price:        An initial price
        """
        self.latest_price = init_price
        self.bucket_volume_array[:] = 0.0
        self.bucket_buy_volume_array[:] = 0.0
        self.bucket_sell_volume_array[:] = 0.0

    def get_current_bucket_idx(self):
        """
        Return the index of the current bucket.
        :rtype:     int
        """
        return self.current_bucket_idx

    def get_previous_bucket_idx(self):
        """
        Return the index of the previous bucket.
        :rtype:     int
        """
        return (self.current_bucket_idx - 1) % self.n_bucket

    def get_bucket_volume(self, idx=None):
        """
        Return bucket_volume_array or its item.
        :params int or None idx:        Index of the bucket to access
        :rtype:                         numpy.ndarray
        """
        if idx is None:
            return self.bucket_volume_array
        else:
            return self.bucket_volume_array[idx]

    def get_bucket_buy_volume(self, idx=None):
        """
        Return buy_volume_array or its item.
        :params int or None idx:        Index of the bucket to access
        :rtype:     numpy.ndarray
        """
        if idx is None:
            return self.bucket_buy_volume_array
        else:
            return self.bucket_buy_volume_array[idx]

    def get_bucket_sell_volume(self, idx=None):
        """
        Return sell_volume_array or its item.
        :params int or None idx:        Index of the bucket to access
        :rtype:     numpy.ndarray
        """
        if idx is None:
            return self.bucket_sell_volume_array
        else:
            return self.bucket_sell_volume_array[idx]

    def get_order_imbalance(self, idx=None):
        """
        Return order_imbalance.
        :params int or None idx:        Index of the bucket to access
        :rtype:     numpy.ndarray or float
        """
        if idx is None:
            return np.abs(
                self.bucket_buy_volume_array
                -
                self.bucket_sell_volume_array
            )
        else:
            return np.abs(
                self.bucket_buy_volume_array[idx]
                -
                self.bucket_sell_volume_array[idx]
            )

    def get_current_bucket_volume_from_last_round(self):
        """
        Return current_bucket_volume_from_last_round.
        :rtype:     numpy.ndarray
        """
        return self.current_bucket_volume_from_last_round

    def get_current_bucket_buy_volume_from_last_round(self):
        """
        Return current_bucket_buy_volume_from_last_round.
        :rtype:     float
        """
        return self.current_bucket_buy_volume_from_last_round

    def get_current_bucket_sell_volume_from_last_round(self):
        """
        Return current_bucket_sell_volume_from_last_round.
        :rtype:     float
        """
        return self.current_bucket_sell_volume_from_last_round

    def move_to_next_bucket(self):
        """
        Move current_bucket_idx to the next bucket.
        """
        self.current_bucket_idx = (self.current_bucket_idx + 1) % self.n_bucket

        self.current_bucket_volume_from_last_round = self.bucket_volume_array[self.current_bucket_idx]
        self.current_bucket_buy_volume_from_last_round = self.bucket_buy_volume_array[self.current_bucket_idx]
        self.current_bucket_sell_volume_from_last_round = self.bucket_sell_volume_array[self.current_bucket_idx]

        self.bucket_volume_array[self.current_bucket_idx] = 0.0
        self.bucket_buy_volume_array[self.current_bucket_idx] = 0.0
        self.bucket_sell_volume_array[self.current_bucket_idx] = 0.0

    @staticmethod
    def check_args(price, pnl_volatility_estimate, volume):
        assert isinstance(price, float), \
            'price should be a float'
        assert price >= 0.0, \
            'price should be non-negative'

        assert isinstance(pnl_volatility_estimate, float), \
            'pnl_volatility_estimate should be a float'
        assert pnl_volatility_estimate >= 0.0, \
            'pnl_volatility_estimate should be non-negative'

        assert isinstance(volume, int) or isinstance(volume, float), \
            'volume should be an int or an float'
        assert volume >= 0.0, \
            'volume should be non-negative'

    def update(self, price, pnl_volatility_estimate, volume):
        """
        Update a new time bar to a bucket in this series of buckets.
        :param float price:                         Price of a new time bar
        :param float pnl_volatility_estimate:       Current volatility estimate of the price pnl
        :param int or float volume:                 Volume of a new time bar
        """
        self.check_args(price, pnl_volatility_estimate, volume)

        assert self.latest_price is not None, \
            'RecursiveBulkClassMABuckets has not been initialized. ' \
            'The latest price is None.'

        buy_volume_pct = norm.cdf(
            (price - self.latest_price)
            /
            pnl_volatility_estimate
        )

        time_bar_remaining_volume = volume
        while time_bar_remaining_volume > 0:
            bucket_remaining_volume = (
                    self.bucket_max_volume
                    -
                    self.bucket_volume_array[self.current_bucket_idx]
            )
            if time_bar_remaining_volume < bucket_remaining_volume:
                self.bucket_volume_array[self.current_bucket_idx] += time_bar_remaining_volume
                buy_volume = np.math.floor(
                    buy_volume_pct * time_bar_remaining_volume
                )
                self.bucket_buy_volume_array[self.current_bucket_idx] += buy_volume
                self.bucket_sell_volume_array[self.current_bucket_idx] += time_bar_remaining_volume - buy_volume
                time_bar_remaining_volume = 0
            else:
                self.bucket_volume_array[self.current_bucket_idx] += bucket_remaining_volume
                buy_volume = np.math.floor(
                    buy_volume_pct * bucket_remaining_volume
                )
                self.bucket_buy_volume_array[self.current_bucket_idx] += buy_volume
                self.bucket_sell_volume_array[self.current_bucket_idx] += bucket_remaining_volume - buy_volume
                self.move_to_next_bucket()
                time_bar_remaining_volume -= bucket_remaining_volume

        self.latest_price = price


class RecursiveBulkClassEWMABuckets(Buckets):
    def __init__(self, bucket_max_volume, past_bucket_decay):
        """
        Instantiate an object of Buckets used for volume bucketing and bulk classification in an
        EWMA style.
        :param int or float bucket_max_volume:      Maximum volume for each bucket (bucket size)
        :param float past_bucket_decay:             Decaying factor the past buckets
        """
        assert isinstance(bucket_max_volume, int) or isinstance(bucket_max_volume, float), \
            'bucket_max_volume should be an int'
        self.bucket_max_volume = bucket_max_volume
        assert isinstance(past_bucket_decay, float) and 0.0 <= past_bucket_decay <= 1.0, \
            'past_bucket_decay should be a float'
        self.past_bucket_decay = past_bucket_decay

        self.current_bucket_volume = 0.0
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0

        self.accumulated_bucket_volume = self.bucket_max_volume
        self.accumulated_order_imbalance = 0.0

        self.running_flag = False

        self.latest_price = None

    def is_initialized(self):
        """
        Check if the buckets is initialized..
        :rtype:     bool
        """
        return self.latest_price is not None

    @staticmethod
    def check_initial_args(initial_price_or_prices, pnl_volatility_estimate, initial_volumes):
        assert isinstance(initial_price_or_prices, float) or isinstance(initial_price_or_prices, np.ndarray), \
            'initial_price_or_prices should be a numpy.ndarray or a float'
        if isinstance(initial_price_or_prices, float):
            assert initial_price_or_prices >= 0.0, \
                'initial_price_or_prices should be non-negative if it is a float'
        assert isinstance(pnl_volatility_estimate, float) or pnl_volatility_estimate is None, \
            'pnl_volatility_estimate should be a float or None'
        if isinstance(pnl_volatility_estimate, float):
            assert pnl_volatility_estimate >= 0.0, \
                'pnl_volatility_estimate should be non-negative if it is a float'
        assert isinstance(initial_volumes, np.ndarray) or initial_volumes is None, \
            'initial_vol should be a numpy.ndarray or None'

    def initialize(self, initial_price_or_prices, pnl_volatility_estimate=None, initial_volumes=None):
        """
        Initialize the buckets in the bulk classification.
        1.  Provide an initial price
        2.  Provide a series of prices in a numpy.ndarray, a volatility estimate and a series of
            volumes in a numpy.ndarray.
        :param float or np.ndarray initial_price_or_prices:     Prices in time bars
        :param None or float pnl_volatility_estimate:           Current volatility estimate of the price pnl
        :param None or np.ndarray initial_volumes:              Volumes in time bars
        """
        self.check_initial_args(initial_price_or_prices, pnl_volatility_estimate, initial_volumes)
        if isinstance(initial_price_or_prices, np.ndarray):
            self.initialize_with_prices_and_volumes(
                initial_price_or_prices,
                pnl_volatility_estimate,
                initial_volumes
            )
        if isinstance(initial_price_or_prices, float):
            self.initialize_with_price(initial_price_or_prices)

    @staticmethod
    def check_prices_and_volumes(initial_prices, initial_volumes):
        assert len(initial_prices) >= 1, \
            'initial_prices should have more than one price'
        assert len(initial_prices.shape) == 1, \
            'initial_prices should be one-dimensional vector'
        assert initial_prices.dtype == 'float64' or initial_prices.dtype == 'float32', \
            'the datatype of initial_prices should be float32 or float64'
        assert len(initial_volumes) >= 1, \
            'initial_volumes should have more than one price'
        assert len(initial_volumes.shape) == 1, \
            'initial_volumes should be one-dimensional vector'
        assert initial_volumes.dtype == 'float64' or initial_volumes.dtype == 'float32', \
            'the datatype of initial_volumes should be float32 or float64'

    def initialize_with_prices_and_volumes(self, initial_prices, pnl_volatility_estimate, initial_volumes):
        """
        Initialize the buckets with an initial price for delta_p in the bulk classification.
        :param np.ndarray initial_prices:           Prices in time bars
        :param float pnl_volatility_estimate:       Current volatility estimate of the price pnl
        :param np.ndarray initial_volumes:          Volumes in time bars
        :return:
        """
        self.initialize_with_price(initial_prices.mean())
        for price, volume in zip(initial_prices, initial_volumes):
            self.update(price, pnl_volatility_estimate, volume)

    def initialize_with_price(self, init_price):
        """
        Initialize the buckets with an initial price for delta_p in the bulk classification.
        :param float init_price:        An initial price
        """
        self.latest_price = init_price

        self.current_bucket_volume = 0.0
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0

        self.accumulated_order_imbalance = 0.0

        self.running_flag = False

    def get_bucket_volume(self):
        """
        Return accumulated_bucket_volume.
        :rtype:     float
        """
        return self.accumulated_bucket_volume

    def get_order_imbalance(self):
        """
        Return order_imbalance.
        :rtype:     numpy.ndarray or float
        """
        return self.accumulated_order_imbalance

    def move_to_next_bucket(self):
        """
        Move current_bucket_idx to the next bucket.
        """

        if self.running_flag:
            self.accumulated_order_imbalance = (
                    (1.0 - self.past_bucket_decay)
                    *
                    np.abs(
                        self.current_bucket_buy_volume
                        -
                        self.current_bucket_sell_volume
                    )
                    +
                    self.past_bucket_decay
                    *
                    self.accumulated_order_imbalance
            )
        else:
            self.accumulated_order_imbalance = np.abs(
                self.current_bucket_buy_volume
                -
                self.current_bucket_sell_volume
            )
            self.running_flag = True

        self.current_bucket_volume = 0.0
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0

    @staticmethod
    def check_args(price, pnl_volatility_estimate, volume):
        assert isinstance(price, float), \
            'price should be a float'
        assert price >= 0.0, \
            'price should be non-negative'

        assert isinstance(pnl_volatility_estimate, float), \
            'pnl_volatility_estimate should be a float'
        assert pnl_volatility_estimate >= 0.0, \
            'pnl_volatility_estimate should be non-negative'

        assert isinstance(volume, int) or isinstance(volume, float), \
            'volume should be an int or an float'
        assert volume >= 0.0, \
            'volume should be non-negative'

    def update(self, price, pnl_volatility_estimate, volume):
        """
        Update a new time bar to a bucket in this series of buckets.
        :param float price:                         Price of a new time bar
        :param float pnl_volatility_estimate:       Current volatility estimate of the price pnl
        :param int or float volume:                 Volume of a new time bar
        """
        self.check_args(price, pnl_volatility_estimate, volume)

        assert self.latest_price is not None, \
            'RecursiveBulkClassMABuckets has not been initialized. ' \
            'The latest price is None.'

        buy_volume_pct = norm.cdf(
            (price - self.latest_price)
            /
            pnl_volatility_estimate
        )

        time_bar_remaining_volume = volume
        while time_bar_remaining_volume > 0:
            bucket_remaining_volume = (
                    self.bucket_max_volume
                    -
                    self.current_bucket_volume
            )
            if time_bar_remaining_volume < bucket_remaining_volume:
                self.current_bucket_volume += time_bar_remaining_volume
                buy_volume = np.math.floor(
                    buy_volume_pct * time_bar_remaining_volume
                )
                self.current_bucket_buy_volume += buy_volume
                self.current_bucket_sell_volume += time_bar_remaining_volume - buy_volume
                time_bar_remaining_volume = 0
            else:
                self.current_bucket_volume += bucket_remaining_volume
                buy_volume = np.math.floor(
                    buy_volume_pct * bucket_remaining_volume
                )
                self.current_bucket_buy_volume += buy_volume
                self.current_bucket_sell_volume += bucket_remaining_volume - buy_volume
                self.move_to_next_bucket()
                time_bar_remaining_volume -= bucket_remaining_volume

        self.latest_price = price
