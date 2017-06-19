import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False


def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True


def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False


_weights_stdev = None


def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev


def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def Linear(name, input_dim, output_dim, inputs, biases=True, initialization=None, weightnorm=None,
           gain=1.):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    with tf.name_scope(name):

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        if initialization == 'glorot' or (initialization is None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        else:

            raise Exception('Invalid initialization!')

        weight_values *= gain

        weight = lib.param(
            name + '.W',
            weight_values
        )

        if weightnorm is None:
            weightnorm = _default_weightnorm

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            result = tf.nn.bias_add(
                result,
                lib.param(
                    name + '.b',
                    np.zeros((output_dim,), dtype='float32')
                )
            )

        return result
