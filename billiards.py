# Piecewise-constant Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import torch

from .utils import ObjectView, to_pickle, from_pickle
from .plotting import coords2images


##### CODE FOR MAKING A DATASET #####

def get_dataset_args(as_dict=False):
    arg_dict = {'num_samples': 10000,
                'train_split': 0.9,
                'time_steps': 5 * 45,
                'num_balls': 2,
                'r': 1e-1,
                'dt': 1e-2,
                'image_every': 5,
                'seed': 0,
                'make_1d': True,
                'verbose': True,
                'use_pixels': False}
    return arg_dict if as_dict else ObjectView(arg_dict)

def make_dataset(args, **kwargs):
  if args.use_pixels and args.verbose:
    print('When Sam profiled this code, it took ~0.35 sec/trajectory.')
    print('\t-> Expect it to take ~1 hr to generate a dataset of 10k samples.')
    import matplotlib.pyplot as plt
    
  np.random.seed(args.seed)
  trajectories, pix_trajectories = [], []
  for i in range(args.num_samples):
    xs = simulate_balls(args.r, args.dt, args.time_steps, args.num_balls, make_1d=args.make_1d, verbose=False)
    xs = xs[::args.image_every]  # subsample the sequence
    if args.use_pixels:
      fig = plt.figure(figsize=(.7,.7), dpi=50)
      pix_xs = coords2images(xs, args.r, process=True, fig=fig)
      pix_trajectories.append(pix_xs)
    trajectories.append(xs)
    if args.verbose:
      print('\rdataset {:.3f}% built'.format(i/args.num_samples * 100), end='', flush=True)

  xs = np.stack(trajectories)
  xs = xs.reshape(*xs.shape[:2], -1).transpose(1,0,2)

  split_ix = int(args.num_samples*args.train_split) # train / test split
  dataset = {'x': xs[:, :split_ix], 'x_test': xs[:, split_ix:], \
             'x_coords': xs[:, :split_ix], 'x_test_coords': xs[:, split_ix:], \
             'dt': args.dt, 'r': args.r, 'num_balls': args.num_balls}
  
  if args.use_pixels:
    pix_xs = np.stack(pix_trajectories)
    pix_xs = pix_xs.reshape(*pix_xs.shape[:2], -1).transpose(1,0,2)
    dataset['x'] = pix_xs[:, :split_ix]
    dataset['x_test'] = pix_xs[:, split_ix:]
  return dataset


# we'll cache the dataset so that it doesn't have to be rebuild every time
def get_dataset(args, path=None, regenerate=False, **kwargs):
    path = './bounce_dat.pkl' if path is None else path
    try:
      if regenerate:
          raise ValueError("Regenerating dataset") # yes this is hacky
      dataset = from_pickle(path)
      if args.verbose:
          print("Successfully loaded data from {}".format(path))
    except:
      if args.verbose:
          print("Did or could not load data from {}. Rebuilding dataset...".format(path))
      dataset = make_dataset(args, **kwargs)
      to_pickle(dataset, path)
    return dataset


##### CODE FOR SIMULATING THE DYNAMICS OF COLLIDING BALLS #####

def rotation_matrix(theta):  # contruct a rotation matrix
  return np.asarray([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

def rotate(x, theta):  # rotate vector x by angle theta
  R = rotation_matrix(theta)
  return (x.reshape(1,-1) @ R)[0]

def angle_between(v0, v1):  # the angle between two vectors
  return np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))

def reflect(x, axis):  # reflect vector x about some other vector 'axis'
  new_xs = np.zeros_like(x)
  for i in range(x.shape[0]):
    theta = angle_between(x[i], axis[i])
    if np.abs(theta) > np.pi/2:
      theta = theta + np.pi
    new_xs[i] = rotate(-x[i], 2 * -theta)
  return new_xs
  
def collide_walls(xs, vs, r, dt):
  mask_low = np.where(xs < r)  # coordinates that are too low
  mask_high = np.where(xs > 1-r)  # coordinates that are too high
  vs[mask_low] *= -1  # rebound
  vs[mask_high] *= -1
  xs[mask_low] = 2*r - xs[mask_low]  # account for overshooting the wall
  xs[mask_high] = (1-r) - (xs[mask_high] - (1-r))
  return xs, vs

def find_colliding_balls(xs, r):
  dist_matrix = ((xs[:,0:1] - xs[:,0:1].T)**2 + (xs[:,1:2] - xs[:,1:2].T)**2)**.5
  dist_matrix[np.tril_indices(xs.shape[0])] = np.inf  # we only care about upper triangle
  body1_mask, body2_mask = np.where(dist_matrix < 2*r)  # select indices of colliding balls
  return body1_mask, body2_mask

def collide_balls(new_xs, vs, r, dt):
  body1_mask, body2_mask = find_colliding_balls(new_xs, r)
  
  # if at least one pair of balls are colliding
  if len(body1_mask) > 0:
    radii_diff = new_xs[body2_mask] - new_xs[body1_mask]  # diff. between radii

    prev_xs = new_xs - vs * dt  # step backward in time
    prev_radii_diff = prev_xs[body2_mask] - prev_xs[body1_mask]

    # if the pair of balls are getting closer to one another
    if np.sum(radii_diff**2) < np.sum(prev_radii_diff**2):
      vs_body1, vs_body2 = vs[body1_mask], vs[body2_mask]  # select the two velocities
      v_com = (vs_body1 + vs_body2) / 2   # find the velocity of the center of masses (assume m1=m2)
      vrel_body1 = vs_body1 - v_com  # we care about relative velocities of the ball

      reflected_vrel_body1 = reflect(vrel_body1, radii_diff)
      vs[body1_mask] = reflected_vrel_body1 + v_com  # rotate velocities (assumes m1=m2)
      vs[body2_mask] = -reflected_vrel_body1 + v_com # symmetry of a perfect collision

  return new_xs, vs

def init_balls(r, num_balls=3, make_1d=False):
  x0 = np.random.rand(num_balls, 2) * (1-2*r) + r  # balls go anywhere in box
  v0 = np.random.randn(*x0.shape)
  if make_1d:
    x0[:,0] = 0.5 ; v0[:,0] = 0  # center and set horizontal velocity to 0
  v0 /= np.linalg.norm(v0, axis=1, keepdims=True)  # velocities start out normalized
  mask, _ = find_colliding_balls(x0, r)  # recursively re-init if any balls overlap
  return init_balls(r, num_balls, make_1d) if len(mask) > 0 else (x0, v0)

def simulate_balls(r=8e-2, dt=2e-2, num_steps=50, num_balls=2, init_state=None, make_1d=False, verbose=False):
  (x0, v0) = init_balls(r, num_balls, make_1d) if init_state is None else init_state
  # x0, v0 = np.flip(x0, axis=0), np.flip(v0, axis=0)  # debugging: simulation should be invariant to this

  curr_x, curr_v = x0, v0
  xs, vs = [x0.copy()], [v0.copy()]
  if verbose: print('initial energy: ', (curr_v**2).sum())
  for i in range(num_steps-1):
    new_xs = xs[-1] + curr_v * dt
    new_xs, curr_v = collide_walls(new_xs, curr_v, r, dt)
    new_xs, curr_v = collide_balls(new_xs, curr_v, r, dt)
    xs.append(new_xs.copy())
    vs.append(curr_v.copy())
  if verbose: print('final energy: ', (curr_v**2).sum())
  return np.stack(xs)