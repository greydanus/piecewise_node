# Piecewise-linear Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import torch

def sample(model, x, autoregress_from, num_steps=None, jumpy=True, constant_dt=None, aux_dims=0, **kwargs):
  if num_steps is None:
    num_steps = x.shape[0]  # time_dim, batch_dim, state_dim = x.shape

  dt_i_prev = 1
  h_i = torch.zeros(x.shape[1], model.rnn_dim, device=x.device)  # init cell state
  x_hat = [x[0]]
  t_hat = [torch.zeros_like(x[0,...,:1]) ]
  h_hat = [h_i]
  
  for i in range(num_steps):
    x_i = x[i] if i < autoregress_from else x_hat[-1]  # forcing vs autoregressive
    z_i = model.encode(x_i)
    h_jumpy_prev = model.do_linear_dynamics(h_i, dt_i_prev, jumpy=jumpy)
    h_i = model.rnn(z_i, h_jumpy_prev)  #, **kwargs

    dt_i = 1
    if i >= autoregress_from-1:
      if jumpy:
        dt_i = model.predict_dt(h_i)
      if constant_dt is not None:
        dt_i = constant_dt

    h_jumpy = model.do_linear_dynamics(h_i, dt_i, jumpy=jumpy)
    dx_i = model.decode(h_jumpy)

    new_x = x_i + dx_i
    if aux_dims > 0:
      new_x[...,-aux_dims:] = 0 * new_x[...,-aux_dims:]

    x_hat.append(new_x)            # store state prediction
    t_hat.append(t_hat[-1] + dt_i) # store time prediction
    h_hat.append(h_i)              # store velocity prediction
    dt_i_prev = dt_i

  x_hat, t_hat, h_hat = [torch.stack(v) for v in [x_hat[:-1], t_hat[:-1], h_hat[1:] ]]
  return x_hat, t_hat, h_hat

def interpolate(model, x_hat, t_hat, h_hat, tvals, offset=0):
  '''A note on `offset`: the model can't handle delta_ts that are less than one.
  Using the `offset` ensures that the model never sees dts less than `offset`.'''
  xvals = []
  for tval in tvals:
    floor_ixs = ((t_hat[...,0] + offset) < tval).sum(0) - 1  # assumes t_hat always increases along time dimension
    floor_ixs = floor_ixs.clamp(0,None)
    floor_xs, floor_ts, floor_hs = [v[floor_ixs, range(len(floor_ixs))]
                                      for v in [x_hat, t_hat, h_hat]]
    delta_ts = (tval - floor_ts).reshape(-1,1)
    h_jumpy = model.do_linear_dynamics(floor_hs, delta_ts)
    delta_xs = model.decode(h_jumpy)
    xvals.append(floor_xs + delta_xs)
  return torch.stack(xvals)

def auto_mse(model, x, args, offset=0, **kwargs):
  x_hat, t_hat, h_hat = sample(model, x, jumpy=args.jumpy, autoregress_from=args.dilate_from, \
                                aux_dims=args.aux_dims, **kwargs)  # sample
  if args.jumpy:
    t_interp = torch.arange(0, x.shape[0])
    x_hat_interp = interpolate(model, x_hat, t_hat, h_hat, t_interp, offset)  # interpolate
    x_hat[args.dilate_from:] = x_hat_interp[args.dilate_from:]
  Ldx = (x - x_hat).pow(2)
  if args.aux_dims > 0:
    Ldx = Ldx[...,:-args.aux_dims]
  return Ldx.mean().item()   # return MSE along trajectory
