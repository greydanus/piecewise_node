Piecewise-linear Neural ODEs
=======
Sam Greydanus, Stefan Lee, Alan Fern | 2021

Summary
--------
Neural networks are a popular tool for modeling sequential data but they generally do not treat time as a continuous variable. Neural ODEs represent an important exception: they parameterize the time derivative of a hidden state with a neural network and then integrate over arbitrary amounts of time. But these parameterizations, which have arbitrary curvature, can be hard to integrate and thus train and evaluate. In this paper, we propose making a piecewise-constant approximation to Neural ODEs to mitigate these issues. Our model can be integrated exactly via Euler integration and can generate autoregressive samples in 3-20 times fewer steps than comparable RNN and ODE-RNN models. We evaluate our model on several synthetic physics tasks and a planning task inspired by the game of billiards. We find that it matches the performance of baseline approaches while requiring less time to train and evaluate.

![hero.png](static/hero.png)

Run in your browser
--------

* Lines experiment ([make dataset](https://colab.research.google.com/drive/11Erg10kBjoaM_92VW7myeDUOZILeLiU0?usp=sharing)) (train jumpy and baseline) (analysis)
* Circles experiment (make dataset) (train jumpy) (train baseline) (analysis)
* Billiards1D experiment (make dataset) (train jumpy) (train baseline) (analysis)
* Billiards2D experiment (make dataset) (train jumpy) (train baseline) (analysis)
* PixelBilliards1D experiment (make dataset) (train jumpy) (train baseline) (analysis)
* PixelBilliards2D experiment (make dataset) (train jumpy) (train baseline) (analysis)


Dependencies
--------
 * NumPy
 * SciPy
 * PyTorch
 * Matplotlib