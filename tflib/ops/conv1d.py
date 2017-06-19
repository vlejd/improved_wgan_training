import tflib as lib

import numpy as np
import tensorflow as tf


_default_weightnorm = False


def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True


def Conv1D(
        name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1,
        weightnorm=None, biases=True, gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, width)
    """
    with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size
        fan_out = output_dim * filter_size / stride

        filters_stdev = np.sqrt(4./(fan_in+fan_out))

        filter_values = uniform(
            filters_stdev,
            (filter_size, input_dim, output_dim)
        )
        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)

        if weightnorm is None:
            weightnorm = _default_weightnorm

        result = tf.nn.conv1d(
            value=inputs,
            filters=filters,
            stride=stride,
            padding='SAME',
            data_format='NCHW'
        )

        return result
