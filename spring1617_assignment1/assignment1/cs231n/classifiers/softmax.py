import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_class = W.shape[1]
  for i in range(num_train):
    score = X[i].dot(W)
    f_i = np.exp(score)
    sum_i = np.sum(f_i)
    for j in range(num_class):
      dW[:, j] += 1.0 / sum_i * f_i[j] * X[i]
      if j == y[i]:
        dW[:, j] -= X[i]
    prob = f_i[y[i]] / sum_i
    loss += -np.log(prob)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_class = W.shape[1]

  score = X.dot(W)
  f = np.exp(score)
  f = f[range(num_train), y] / np.sum(f, axis=1)
  loss = np.sum(-np.log(f))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #calculate the probability of each class
  p_k = np.exp(score)
  p_k = p_k / np.sum(p_k, axis=1, keepdims=True)
  #substract 1 in the correct class, according to the formula dL_i = (p_k - 1(y_i = k))x_i
  p_k[range(num_train), y] -= 1
  dW += X.T.dot(p_k)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

