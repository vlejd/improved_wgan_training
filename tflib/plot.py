import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os

import collections
import cPickle as pickle

matplotlib.use('Agg')
_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = 0
_dir = '.'


def set_dir(dir):
    global _dir
    _dir = dir


def tick():
    global _iter
    _iter += 1


def plot(name, value):
    _since_last_flush[name][_iter] = value


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(vals.values())))
        _since_beginning[name].update(vals)

        x_vals = np.sort(_since_beginning[name].keys())
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(_dir, name.replace(' ', '_')+'.jpg'))

    print("iter {}\t{}".format(_iter, "\t".join(prints)))
    _since_last_flush.clear()

    with open(os.path.join(_dir, 'log.pkl'), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
