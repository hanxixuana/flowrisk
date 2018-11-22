#!/usr/bin/env python
#
# Created by Xixuan on Nov 20, 2018
#

import pandas as pd

from abc import ABCMeta, abstractmethod


class Measure:

    __metaclass__ = ABCMeta


class RecursiveMeasure(Measure):

    @abstractmethod
    def update(self, time_bar):
        """
        Update the measure using a new time bar.
        :param pd.Series time_bar:      New time bar of the market information
        :rtype:                         float
        """
        raise NotImplementedError


class BulkMeasure(Measure):

    @abstractmethod
    def estimate(self, data):
        """
        Estimate the measure using time series data.
        :param pd.DataFrame data:       DataFrame of market data
        :rtype:                         pd.Series
        """
        raise NotImplementedError
