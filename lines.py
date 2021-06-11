# Piecewise-constant Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import torch

from .utils import ObjectView


def get_dataset_args(as_dict=False):
  arg_dict = {'num_samples': 10000,
              'train_split': 0.9,
              'dt': 0.1,
              'seed': 0}
  return arg_dict if as_dict else ObjectView(arg_dict)


def make_line(dt=0.1, domain=[0, 2.1]):
  start, stop = domain
  xvals = np.linspace(start, stop, int((stop-start)/dt))
  yvals = np.ones_like(xvals) * np.random.rand()
  return np.stack([xvals, yvals])
  
def get_dataset(args):
  np.random.seed(args.seed)  # random seed for reproducibility
  trajectories = []

  print(args.num_samples)
  for i in range(args.num_samples):  # this loop generates the synthetic 'circles' dataset
    x = make_line(dt=args.dt)
    trajectories.append( x )  # append trajectory to list

  seq = np.stack(trajectories).transpose(2,0,1)  # reshape tensor dimensions -> [time, batch, state]
  split_ix = int(args.num_samples*args.train_split) # train / test split
  dataset = {'x': seq[:, :split_ix], 'x_test': seq[:, split_ix:]}
  return dataset