# Piecewise-constant Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime

from .utils import ObjectView


##### CODE FOR MAKING A DATASET #####
  
def get_dataset_args(as_dict=False):
    arg_dict = {'url': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
                'time_steps': 30,
                'subsample_rate': 6,
                'train_split': 0.8,
                'seed': 42,
                'verbose': True}
    return arg_dict if as_dict else ObjectView(arg_dict)

def get_dataset(args=None):
  if args is None:
    args = get_dataset_args()
  np.random.seed(args.seed)

  r = requests.get(args.url, allow_redirects=True)
  open('./data.csv.zip', 'wb').write(r.content)
  data = pd.read_csv('./data.csv.zip')
  data = data[args.subsample_rate-1::args.subsample_rate]

  date_time = pd.to_datetime(data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
  timestamp_s = date_time.map(datetime.datetime.timestamp)
  data['day_sin'] = np.sin(timestamp_s * (2 * np.pi / (24*60*60)))
  data['yr_sin'] = np.sin(timestamp_s * (2 * np.pi / (24*60*60*365.2425)))

  variables = ['T (degC)', 'p (mbar)', 'rho (g/m**3)', 'day_sin', 'yr_sin']
  xs = np.stack([np.asarray(data[k]) for k in variables]).T
  xs = (xs - xs.mean(0, keepdims=True)) / xs.std(0, keepdims=True)

  tchunk = args.time_steps
  xs = xs[:tchunk*(len(xs)//tchunk)]
  xs = xs.reshape(-1,tchunk, len(variables))  # shape according to [time, batch]

  shuffle_ixs = np.random.permutation(xs.shape[0])
  xs = xs[shuffle_ixs].transpose(1,0,2)
  split_ix = int(xs.shape[1]*args.train_split) # train / test split
  data = {'x': xs[:, :split_ix], 'x_test': xs[:, split_ix:]}
  return data