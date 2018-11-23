#!/usr/bin/env python
#
#   Created by Xixuan on Nov 20, 2018
#

import numpy as np

from abc import ABCMeta, abstractmethod


class Vol:

    __metaclass__ = ABCMeta

    @abstractmethod
    def initialize(self, initial_price_or_prices, vol):
        """
        Initialize the EWMA estimation process of the sample volatility of prices.
        :param float or np.ndarray initial_price_or_prices:     Array of prices
        :param float vol:                                       Initial value for volatility
        :rtype:                                                 float
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, price):
        """
        Update the EWMA estimate of the sample volatility of prices.
        :param float price:     Latest price
        :rtype:                 float
        """
        raise NotImplementedError


class RecursiveEWMAVol(Vol):
    def __init__(self, decay, value_type):
        """
        Instantiate a recursive EWMA estimator of sample volatility of prices.
        :param float decay:                     Decaying factor for past prices
        :param str value_type:                  'pnl' or 'simple' or 'log'
        """
        assert isinstance(decay, float) and 0.0 <= decay <= 1.0, \
            'decay should be a float in [0.0, 1.0]'
        self.decay = decay
        assert value_type in ('pnl', 'simple', 'log'), \
            'value_type should be "pnl" or "simple" or "log"'
        self.value_type = value_type

        self.latest_price = None
        self.latest_var = None

    def is_initialized(self):
        """
        Check if the recursive estimator is initialized. Since the estimator is ready to work
        simply after an initial price is given, we use it to do the check. Note that, at the
        beginning, the volatility estimates would be quite unstable if only a price is given
        to initialize the recursive estimator.
        :rtype:     bool
        """
        return self.latest_price is not None

    @staticmethod
    def check_initial_args(initial_price_or_prices, initial_vol):
        assert isinstance(initial_price_or_prices, float) or isinstance(initial_price_or_prices, np.ndarray), \
            'initial_price_or_prices should be a float or a numpy.ndarray'
        if isinstance(initial_price_or_prices, float):
            assert initial_price_or_prices >= 0.0, \
                'initial_price_or_prices should be non-negative if it is a float'
        assert isinstance(initial_vol, float) or initial_vol is None, \
            'initial_vol should be a float or None'
        if isinstance(initial_vol, float):
            assert initial_vol >= 0.0, \
                'initial_vol should be non-negative if it is a float'

    def initialize(self, initial_price_or_prices, initial_vol=None):
        """
        There are two ways to do the initialization.
        1.  Provide an initial price and an initial volatility
        2.  Provide a series of prices in a numpy.ndarray.
        :param float or np.ndarray initial_price_or_prices:     A single price of a series of prices
        :param float or None initial_vol:                       A single initial volatility
        :rtype:                                                 float
        """
        self.check_initial_args(initial_price_or_prices, initial_vol)
        if isinstance(initial_price_or_prices, float):
            self.initialize_with_price_and_vol(initial_price_or_prices, initial_vol)
        if isinstance(initial_price_or_prices, np.ndarray):
            self.initialize_with_prices(initial_price_or_prices)

    def initialize_with_price_and_vol(self, init_price, init_vol):
        """
        Initialize the recursive estimator with a price and/or a volatility
        :param float init_price:                Initializing price
        :param float or None init_vol:          Initializing volatility
        """
        self.latest_price = init_price
        self.latest_var = init_vol ** 2.0 if init_vol is not None else init_vol

    @staticmethod
    def check_prices(prices):
        assert len(prices) > 1, \
            'prices should have more than one price'
        assert len(prices.shape) == 1, \
            'prices should be one-dimensional vector'
        assert prices.dtype == 'float64' or prices.dtype == 'float32', \
            'the datatype of prices should be float32 or float64'

    def initialize_with_prices(self, prices):
        """
        Initialize the EWMA estimation process of the sample volatility of prices.
        :param np.ndarray prices:       Array of prices
        """
        self.check_prices(prices)

        if self.value_type == 'pnl':
            squared_log_returns = np.square(
                prices[1:] - prices[:-1]
            )
        elif self.value_type == 'simple':
            squared_log_returns = np.square(
                prices[1:] / prices[:-1] - 1.0
            )
        elif self.value_type == 'log':
            squared_log_returns = np.square(
                np.log(prices[1:] / prices[:-1])
            )
        else:
            raise ValueError('return_type should be "simple" or "log"')
        accumulator = (
            lambda prev_ret, curr_ret: self.decay * prev_ret + (1.0 - self.decay) * curr_ret
        )
        self.latest_price = prices[-1]
        self.latest_var = reduce(accumulator, squared_log_returns)

    def check_price(self, price):
        assert isinstance(price, float), \
            'price should be float'
        assert price >= 0.0, \
            'price should be non-negative'
        assert self.latest_price is not None, \
            'Has not initialized the initial price'

    def update(self, price):
        """
        Update the EWMA estimate of the sample volatility of prices.
        :param float price:     Latest price
        :rtype:                 float
        """
        self.check_price(price)

        assert self.latest_price is not None, \
            'RecursiveEWMAVol has not been initialized. ' \
            'The latest price is None.'

        if self.value_type == 'pnl':
            if self.latest_var is not None:
                self.latest_var = (
                        (1.0 - self.decay) * (price - self.latest_price) ** 2.0
                        +
                        self.decay * self.latest_var
                )
            else:
                self.latest_var = (price - self.latest_price) ** 2.0
        elif self.value_type == 'simple':
            if self.latest_var is not None:
                self.latest_var = (
                        (1.0 - self.decay) * (price / self.latest_price - 1.0) ** 2.0
                        +
                        self.decay * self.latest_var
                )
            else:
                self.latest_var = (price / self.latest_price - 1.0) ** 2.0
        elif self.value_type == 'log':
            if self.latest_var is not None:
                self.latest_var = (
                        (1.0 - self.decay) * np.math.log(price / self.latest_price) ** 2.0
                        +
                        self.decay * self.latest_var
                )
            else:
                self.latest_var = np.math.log(price / self.latest_price) ** 2.0
        else:
            raise ValueError('return_type should be "simple" or "log"')
        self.latest_price = price
        return np.math.sqrt(self.latest_var)

    def get_latest_vol(self):
        return np.math.sqrt(self.latest_var)


def test():

    accuracy = 1e-8
    vol = RecursiveEWMAVol(0.7, value_type='log')

    prices = np.ones(20) * 5.0
    vol.initialize(prices)
    assert np.abs(vol.get_latest_vol() - 0.0) < accuracy

    prices = np.exp(np.arange(1.0, 10.0, 1.0))
    vol.initialize(prices)
    assert np.abs(vol.get_latest_vol() - 1.0) < accuracy

    prices = np.exp(np.arange(1.0, 10.0, 0.25))
    vol.initialize(prices)
    assert np.abs(vol.get_latest_vol() - 0.25) < accuracy

    vol.initialize(np.math.exp(0.5))
    prices = np.exp(np.arange(2.0, 100.0, 1.0))
    for price in prices:
        vol.update(price)
    assert np.abs(vol.get_latest_vol() - 1.0) < accuracy

    vol.initialize(np.math.exp(1.5))
    prices = np.exp(np.arange(2.0, 100.0, 1.0))
    for price in prices:
        vol.update(price)
    assert np.abs(vol.get_latest_vol() - 1.0) < accuracy

    vol.initialize(np.math.exp(0.5), 1.5)
    prices = np.exp(np.arange(2.0, 100.0, 1.0))
    for price in prices:
        vol.update(price)
    assert np.abs(vol.get_latest_vol() - 1.0) < accuracy

    vol.initialize(np.math.exp(0.5), 0.5)
    prices = np.exp(np.arange(2.0, 100.0, 1.0))
    for price in prices:
        vol.update(price)
    assert np.abs(vol.get_latest_vol() - 1.0) < accuracy

    print('all passed')
