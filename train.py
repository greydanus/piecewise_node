# Piecewise-constant Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import time, copy
import torch

from .teacher_forcing import teacher_forcing
from .utils import ObjectView
from.sample import auto_mse

def get_train_args(as_dict=False):
  arg_dict = {'input_dim': 2,
              'rnn_dim': 128,
              'hidden_dim': 128,
              'learning_rate': 1e-3,
              'gamma': 0.75,
              'decay_lr_every': 1000,
              'epsilon': 3e-3,
              'dt_loss_coeff': 1e-5,
              'jumpy': True,
              'jump_prob': 1e-2,
              'weight_decay': 1e-7,
              'aux_dims': 2,  # auxiliary input dimensions that we won't include in the loss
              'batch_size': 256,
              'total_steps': 25000,
              'print_every': 500,
              'eval_every': 500,
              'dilate_from': 4,
              'checkpoint_every': 1000000,
              'seed': 0,
              'device': 'cuda'}
  return arg_dict if as_dict else ObjectView(arg_dict)


def get_batch(v, i, args):  # helper function for moving batches of data to/from GPU
  bix = (i*args.batch_size) % v.shape[1]
  return v[:, bix:bix + args.batch_size].to(args.device)
  
def train(args, model, data, **kwargs):
  model.train().to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr_every, gamma=args.gamma)
  results = {k:[] for k in ['Ldx', 'Ldt', 'Ldx_test', 'Ldt_test', 'mean_dt', 'mean_dt_test', \
                            'walltime', 'areg_error', 'areg_steps', 'best_model', 'auto_mse']}
  results['best_model'] = {'test_loss': np.inf, 'model': None}
  t0 = time.time()
  for step in range(args.total_steps+1):

    seq = get_batch(data['x'], step, args)
    x, dx = seq[:-1], seq[1:] - seq[:-1]
    Ldx, Ldt, mean_dt = teacher_forcing(model, x, dx, args, jumpy=args.jumpy)  # main training step

    loss = Ldx + Ldt  # backprop and logging
    loss.backward() ; optimizer.step() ; optimizer.zero_grad() ; scheduler.step()
    results['Ldx'].append(Ldx.item())
    results['Ldt'].append(Ldt.item())
    results['mean_dt'].append(mean_dt)

    if args.eval_every > 0 and step % args.eval_every == 0:     # evaluate the model
        seq = data['x_test'].to(args.device)                    # test set -> GPU
        x_, dx_ = seq[:-1], seq[1:] - seq[:-1]
        Ldx_, Ldt_, mean_dt_ = teacher_forcing(model, x_, dx_, args, jumpy=args.jumpy)
        results['Ldx_test'].append(Ldx_.item())  # dx test loss
        results['Ldt_test'].append(Ldt_.item())  # dt test loss
        results['mean_dt_test'].append(mean_dt_)    # mean dt (over test data)
        results['auto_mse'].append(auto_mse(model, x_, args, offset=.5))

    if step > 0 and step % args.print_every == 0:   # print out training progress
        t1 = time.time()
        results['walltime'].append(t1-t0) ; t0 = t1
        metrics = [results[k][-1] for k in ['walltime', 'Ldx', 'Ldx_test', 'Ldt', 'mean_dt', 'mean_dt_test', 'auto_mse']]
        print("step {}, dt {:.2f}s, Ldx {:.2e}, Ldx_test {:.2e}, Ldt {:.2e}, " \
              "mean_dt {:.2f}, mean_dt_test {:.2f}, auto_mse {:.2e}".format(step, *metrics))

  results['last_model'] = model.cpu() #copy.deepcopy(model).cpu()  # finish logging
  if model.use_ode:
    results['last_model'] = model.cpu().state_dict()
  results['args'] = copy.deepcopy(args)
  return results
