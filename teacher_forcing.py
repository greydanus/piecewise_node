# Piecewise-constant Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import torch

from .utils import masked_mse

def teacher_forcing(*args, jumpy=False, **kwargs):
  if jumpy:
    return jumpy_teacher_forcing(*args, **kwargs)
  return normal_teacher_forcing(*args, **kwargs)


def normal_teacher_forcing(model, x, dx, args, **kwargs):
  z = model.encode(x)
  time_dim, batch_dim, rnn_dim = z.shape  # get dimensions of data tensor

  h_i = torch.zeros(batch_dim, args.rnn_dim, device=z.device)  # init cell state
  # potentially add start token
  hs = []
  for i in range(time_dim):
    h_i = model.rnn(z[i], h_i, **kwargs)         # GRU cell "tick" prediction
    hs.append(h_i)
  
  Ldt = torch.zeros(1)[0].detach()
  Ldx = (model.decode(torch.stack(hs)) - dx).pow(2)  # calculate dx loss
  if args.aux_dims > 0:
    Ldx = Ldx[...,:-args.aux_dims]
  Ldx = Ldx.mean() #+ (44 * Ldx[3:4]).sum()  # re-weight time step where force was applied
  mean_dt = 1.
  return Ldx, Ldt, mean_dt


def jumpy_teacher_forcing(model, x, dx, args, eps=1e-10, **kwargs):
  z = model.encode(x)
  time_dim, batch_dim, rnn_dim = z.shape  # get dimensions of data tensor

  h_i = torch.zeros(batch_dim, args.rnn_dim, device=z.device)  # init cell state
  dt_i = torch.ones(batch_dim, 1, device=z.device)  # init vector to track delta_ts
  dx_prev = torch.zeros_like(dt_i)                  # init vector to track delta_xs
  
  hs, dxs, dts, final_dts = [], [], [], []
  for i in range(time_dim):
    h_pretick = model.do_linear_dynamics(h_i, dt_i)
    h_tick = model.rnn(z[i], h_pretick, **kwargs)    # GRU cell "tick" prediction
    
    h_jumpy = model.do_linear_dynamics(h_i, dt_i+1)  # jumpy prediction
    dx_jumpy = (dx_prev + dx[i]).detach()           # delta x for jumpy prediction

    if i >= args.dilate_from:
      L_jumpy = (model.decode(h_jumpy) - dx_jumpy).pow(2)
      if args.aux_dims > 0:
        L_jumpy = L_jumpy[...,:-args.aux_dims]
      L_jumpy = L_jumpy.mean(-1, keepdims=True)
      M_pert = torch.rand(batch_dim, 1, device=x.device) < args.jump_prob
      M = ((L_jumpy.detach() < args.epsilon) + M_pert).int()  # decide whether to use jumpy update
    else:
      M = 0
    
    # gate model based on whether a) the update was jumpy or b) the RNN cell 'ticked'
    final_dts.append(dt_i * (1-M))
    h_i = h_i * M  +  h_tick * (1-M)           # if not jumpy, update the cell state
    dt_i = (dt_i+1) * M  +  1. * (1-M)         # advance one time step, or reset delta_t to 1
    dx_prev = dx_jumpy * M  +  dx[i] * (1-M)   # if jumpy, increase the delta in the target data
    hs.append(h_i)  ;  dts.append(dt_i)  ;  dxs.append(dx_prev)
  
  hs, dxs, dts, final_dts = [torch.stack(v) for v in [hs, dxs, dts, final_dts]]  # stack along time dimension
  final_dts = final_dts[args.dilate_from:]     # shift dts back one timestep to make targets
  dt_guesses = model.predict_dt(hs[args.dilate_from-1:-1])

  nonzero_final_dts = final_dts[final_dts>0]
  if len(nonzero_final_dts) > 0:
    mean_dt = nonzero_final_dts.mean().item()
    Ldt = args.dt_loss_coeff * masked_mse(dt_guesses, final_dts, final_dts>0, eps)  # calculate dt loss
  else:
    mean_dt = time_dim
    Ldt = args.dt_loss_coeff * (dt_guesses-time_dim).pow(2).mean()  # jump straight to the end

  next_hs = model.do_linear_dynamics(hs, dts)
  Ldx = ((model.decode(next_hs) - dxs).pow(2) + eps)  # calculate dx loss
  if args.aux_dims > 0:
    Ldx = Ldx[...,:-args.aux_dims]
  Ldx = Ldx.mean() #+ (44 * Ldx[3:4]).sum()  # re-weight time step where force was applied

  current_hs = model.do_linear_dynamics(hs, 0)
  Ldx += 1e-1 * ((model.decode(current_hs)).pow(2) + eps).mean()   # setting dt=0 should give dx=0

  # dts = final_dts[final_dts>0]
  # mean_dt = dts.mean().item() if len(dts) > 0 else time_dim
  return Ldx, Ldt, mean_dt
  