# Piecewise-linear Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchdyn
from torchdyn.models import NeuralDE

##### A SMORGASBORD OF ENCODERS AND DECODERS #####

class IdentityFn(nn.Module):
  def forward(self, x):
    return x


class ResidualMLP(nn.Module):
  def __init__(self, input_dim, target_dim, hidden_dim=64):
    super(ResidualMLP, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = nn.Linear(hidden_dim, target_dim)

  def forward(self, h):
    h = self.linear1(h).relu()
    h = h + self.linear2(h).relu()  # residual connection
    h = h + self.linear3(h).relu()  # residual connection
    return self.linear4(h)


##### THE MAINFRAME SEQUENCE MODEL (RNN CORE + JUMPY ARITHMETIC + SAMPLING APIs) #####

class SequenceModel(nn.Module):
  '''A generic sequence model with an RNN cell at its core. This model has two
  rather different modes: "forward" and "autoregress." The former is used for
  training and the latter is used for sampling.'''
  def __init__(self, input_dim, hidden_dim, rnn_dim, time_constant=0.2, encoder=None, decoder=None, use_ode=False):
    super(SequenceModel, self).__init__()
    if encoder is None:
      encoder = ResidualMLP(input_dim=input_dim, target_dim=hidden_dim, hidden_dim=hidden_dim)
    self.encoder = encoder
    self.rnn = nn.GRUCell(hidden_dim, rnn_dim)
    if decoder is None:
      decoder = ResidualMLP(input_dim=rnn_dim, target_dim=input_dim, hidden_dim=hidden_dim)
    self.decoder = decoder
    self.dt_predictor = nn.Linear(rnn_dim, 1)

    self.input_dim, self.hidden_dim, self.rnn_dim = input_dim, hidden_dim, rnn_dim
    self.time_constant = time_constant
    self.leaky_relu = torch.nn.LeakyReLU()
    
    if use_ode:
      hh = int(rnn_dim/2) #if args.jumpy else hidden_dim
      self.f = nn.Sequential(nn.Linear(hh, hh),
                    nn.Tanh(), nn.Linear(hh, hh))
      self.ode = NeuralDE(self.f, sensitivity='adjoint', solver='rk4')
    self.use_ode = use_ode

  def encode(self, x):
    return self.encoder(x)

  def decode(self, h):
    return self.decoder(h * self.time_constant)

  def predict_dt(self, h):
    dt_hat = self.dt_predictor(h)
    return self.leaky_relu(dt_hat).add(1)  #0.75 + self.dt_predictor(h).exp()

  def do_linear_dynamics(self, h, dt, jumpy=True):
    state_dim = h.shape[-1]
    dynamics_dim = state_dim // 2

    if not jumpy:
      if self.use_ode:
        z, dh0 = h[...,:dynamics_dim], h[...,dynamics_dim:]
        z_next = z + 0.01 * self.ode(dh0)
        h = torch.cat([z_next, dh0], axis=-1)
        return h
      else:
        return h
    
    else:
      z, dzdt = h[...,:dynamics_dim], h[...,dynamics_dim:] * self.time_constant
      z_next = z + dzdt * dt
      h_next = torch.cat([z_next, dzdt], axis=-1)
    return h_next
    