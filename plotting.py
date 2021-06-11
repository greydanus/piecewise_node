# Piecewise-constant Neural ODEs
# Sam Greydanus, Stefan Lee, Alan Fern

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.collections as mcollections

from skimage.transform import resize
from moviepy.editor import ImageSequenceClip


##### GENERIC PLOTTING UTILITIES #####

def fig2image(fig):
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return image


##### CODE SPECIFIC TO 'BILLIARDS' DATASET #####

class UpdatablePatchCollection(mcollections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        mcollections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths
        
def process_images(raw_ims, image_every=1):
  side_len = raw_ims[0].shape[0]
  k, dx, dy = int(0.15*side_len), int(0.05*side_len), int(0.02*side_len)
  new_ims = []
  for im in raw_ims[::image_every]:
    im = im[k+dy:-k+dy,k+dx:-k+dx].mean(-1)
    im = resize(im, (28, 28)) / 255.
    new_ims.append(im)
  return np.stack(new_ims)

def update_plot(fig, x, balls):
  if len(balls) == 2:
    colors = ['#000000', '#000000'] # make both balls black
  else:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  for j, b in enumerate(balls):
    b.set_center((x[j,0], x[j,1]))
    b.set_fc(colors[j])
    fig.canvas.draw()
    fig.canvas.flush_events()

def coords2images(xs, r, fig=None, process=False, figsize=(2,2), dpi=50, **args):
  fig = plt.figure(figsize=figsize, dpi=dpi) if fig is None else fig

  plt.ion()
  balls = [Circle((0,0), r) for _ in range(xs.shape[-2])]
  collection = UpdatablePatchCollection(balls)

  ax = fig.gca()
  ax.add_artist(collection)
  [ax.add_artist(b) for b in balls]
  ax.set_xlim(0, 1) ; ax.set_ylim(0, 1)
  ax.set_aspect('equal', adjustable='box')
  ax.get_xaxis().set_ticks([]) ; ax.get_yaxis().set_ticks([])

  images = []
  for i in range(xs.shape[0]):
    update_plot(fig, xs[i], balls)
    images.append( fig2image(fig) )
  plt.close()
  images = np.stack(images)
  return process_images(images, **args) if process else images

def tensor2videoframes(x):
  n, w, h = x.shape
  frames = []
  for x_i in x:
    x_i = x_i[...,np.newaxis].repeat(3,-1)
    x_i = resize(x_i, (6*w, 6*h), anti_aliasing=False) * 255
    frames.append(x_i)
  return frames