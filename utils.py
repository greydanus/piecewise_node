# Piecewise-linear Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import pickle

class ObjectView(object):  # make a dictionary look like an object
  def __init__(self, d): self.__dict__ = d


def masked_mse(pred, target, mask, eps=1e-10):  # compute MSE over unmasked part of a tensor
  return (pred - target).pow(2).mul(mask).sum().add(eps) / (eps + 1.*mask.sum())


def to_pickle(thing, path):  # save something
  with open(path, 'wb') as handle:
      pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path):  # load something
  thing = None
  with open(path, 'rb') as handle:
      thing = pickle.load(handle)
  return thing