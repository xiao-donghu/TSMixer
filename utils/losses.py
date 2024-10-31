# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import tensorflow as tf
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = tf.math.divide(a, b)
    result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)
    result = tf.where(tf.math.is_inf(result), tf.zeros_like(result), result)
    return result


class mape_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(mape_loss, self).__init__()

    def call(self, insample, freq, forecast, target, mask):
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        loss = tf.reduce_mean(tf.abs((forecast - target) * weights))
        return loss


class smape_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(smape_loss, self).__init__()

    def call(self, insample, freq, forecast, target, mask):
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        denominator = tf.abs(forecast) + tf.abs(target)
        smape = 200 * divide_no_nan(tf.abs(forecast - target), denominator)
        loss = tf.reduce_mean(smape * mask)
        return loss


class mase_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(mase_loss, self).__init__()

    def call(self, insample, freq, forecast, target, mask):
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        # Calculate the MASE denominator (mean absolute scaled error)
        insample_diff = insample[:, freq:] - insample[:, :-freq]
        masep = tf.reduce_mean(tf.abs(insample_diff), axis=1)
        masep_inv = divide_no_nan(mask, tf.expand_dims(masep, axis=1))

        # Calculate the MASE loss
        loss = tf.reduce_mean(tf.abs(target - forecast) * masep_inv)
        return loss
