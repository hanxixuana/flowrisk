#!/usr/bin/env python
#
# Created by Xixuan on Nov 21, 2018
#

import numpy as np

from flowrisk.tools.vol import RecursiveEWMAVol

from abc import ABCMeta, abstractmethod


class Band:

    __metaclass__ = ABCMeta

    @abstractmethod
    def is_initialized(self):
        """
        Check if the recursive estimator is initialized.
        :rtype:     bool
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, initial_value_or_values, initial_log_innovation_vol=None):
        """
        There are two ways to do the initialization.
        1.  Provide an initial value and an initial log innovation volatility
        2.  Provide a series of values in a numpy.ndarray.
        :param float or np.ndarray initial_value_or_values:     A single value or a series of prices
        :param float or None initial_log_innovation_vol:        A single initial log innovation volatility
        :rtype:                                                 float
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, value):
        """
        Update the confidence interval using a new value.
        :param float value:     A new value of the time series
        :rtype:                 tuple
        """
        raise NotImplementedError


class RecursiveEWMABand(Band):
    def __init__(self, mean_decay, vol_decay, band_radius):
        """
        Instantiate an object of a series of EWMA based moving confidence intervals
        for a time series of heterogeneous volatility.
        :param float mean_decay:        Decaying factor for past values for updating the moving mean
        :param float vol_decay:         Decaying factor for past values for updating the moving volatility
        :param float band_radius:       Amount of log innovation volatility for the band radius
        """
        assert isinstance(mean_decay, float) and 0.0 <= mean_decay <= 1.0, \
            'mean_decay should be a float in [0.0, 1.0]'
        self.mean_decay = mean_decay
        assert isinstance(vol_decay, float) and 0.0 <= vol_decay <= 1.0, \
            'vol_decay should be a float in [0.0, 1.0]'
        self.vol_decay = vol_decay
        assert isinstance(band_radius, float) and 0.0 <= band_radius, \
            'band_radius should be a float in [0.0, +infinity]'
        self.band_radius = band_radius

        self.latest_value = None

        self.upper_line_estimate = None
        self.central_line_estimate = None
        self.lower_line_estimate = None

        self.mean_estimate = None
        self.log_innovation_vol_estimator = RecursiveEWMAVol(vol_decay, value_type='log')

    def is_initialized(self):
        """
        Check if the recursive estimator is initialized.
        :rtype:     bool
        """
        return (
                self.central_line_estimate is not None
                and
                self.latest_value is not None
                and
                self.mean_estimate is not None
        )

    def initialize(self, initial_value_or_values, initial_log_innovation_vol=None):
        """
        There are two ways to do the initialization.
        1.  Provide an initial value and an initial log innovation volatility
        2.  Provide a series of values in a numpy.ndarray.
        :param float or np.ndarray initial_value_or_values:     A single value or a series of prices
        :param float or None initial_log_innovation_vol:        A single initial log innovation volatility
        :rtype:                                                 float
        """
        assert isinstance(initial_value_or_values, float) or isinstance(initial_value_or_values, np.ndarray), \
            'initial_value_or_values should be a float or a numpy.ndarray'

        if isinstance(initial_value_or_values, float):
            assert initial_value_or_values >= 0.0, \
                'initial_value_or_values should be non-negative if it is a float'
            self.central_line_estimate = initial_value_or_values
            self.latest_value = initial_value_or_values
            self.mean_estimate = 0.0

        if isinstance(initial_value_or_values, np.ndarray):
            assert len(initial_value_or_values) > 1, \
                'initial_value_or_values should have more than one price'
            assert len(initial_value_or_values.shape) == 1, \
                'initial_value_or_values should be one-dimensional vector'
            assert initial_value_or_values.dtype == 'float64' or initial_value_or_values.dtype == 'float32', \
                'the datatype of initial_value_or_values should be float32 or float64'
            self.central_line_estimate = initial_value_or_values[-1]
            self.latest_value = initial_value_or_values[-1]
            self.mean_estimate = (
                (
                    np.log(
                        initial_value_or_values[1:]
                        /
                        initial_value_or_values[:-1]
                    )
                ).mean()
            )

        self.log_innovation_vol_estimator.initialize(
            initial_value_or_values, initial_log_innovation_vol
        )

    def update(self, value):
        """
        Update the confidence interval using a new value.
        :param float value:     A new value of the time series
        :rtype:                 list
        """

        assert self.mean_estimate is not None, \
            'RecursiveEWMABand has not been initialized. ' \
            'The mean estimate is None.'

        self.mean_estimate = (
                (1.0 - self.mean_decay) * np.math.log(value / self.latest_value)
                +
                self.mean_decay * self.mean_estimate
        )
        self.log_innovation_vol_estimator.update(value)

        self.latest_value = value

        self.central_line_estimate *= np.math.exp(self.mean_estimate)

        deviation = np.math.exp(
            self.band_radius
            *
            self.log_innovation_vol_estimator.get_latest_vol()
        )

        self.upper_line_estimate = self.central_line_estimate * deviation
        self.lower_line_estimate = self.central_line_estimate / deviation

        return [
            self.lower_line_estimate,
            self.central_line_estimate,
            self.upper_line_estimate
        ]


def test():

    import matplotlib.pyplot as plt
    values = np.exp(np.random.randn(1000).cumsum() / 10.0)

    band = RecursiveEWMABand(0.9, 0.9, 2.0)
    band.initialize(values[:200])
    result = list()
    for val in values[200:]:
        result.append(
            band.update(val)
        )
    result = np.array(result)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(values[200:], 'k')
    ax.plot(result[:, 1], 'r', linewidth=2)
    ax.plot(result[:, 0], 'b--')
    ax.plot(result[:, 2], 'b--')
    fig.show()

    band.initialize(values[0], 0.0)
    result = list()
    for val in values:
        result.append(
            band.update(val)
        )
    result = np.array(result)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(values, 'k')
    ax.plot(result[:, 1], 'r', linewidth=2)
    ax.plot(result[:, 0], 'b--')
    ax.plot(result[:, 2], 'b--')
    fig.show()
