import math

import tensorflow as tf
import random
import numpy as np


def create_batches(data_size, batch_size, shuffle=True):
  """create index by batches."""
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
  rest = data_size % batch_size
  if rest > 0:
    batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
  return batches

def fetch_data(data, count, idx_batch, vocab_size):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  mask = np.zeros(batch_size)
  data_batch = np.zeros((batch_size, vocab_size))
  array_idx = np.array(idx_batch)
  actual = array_idx != -1
  data_batch[actual] = data[array_idx[actual]]
  mask[actual] = 1.
  count_batch = count[idx_batch]
  return data_batch, count_batch, mask

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
    elif prefix in varname:
      ret_list.append(var)
  return ret_list


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
   
    if weights is not None:
      matrix=weights
    else:
      matrix = tf.get_variable('Matrix', [input_size, output_size],initializer=matrix_initializer)
    
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term
  return output

def mlp(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=None,
        scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    if mlp_nonlinearity:
      for l in range(mlp_layer):
        res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    else:
      for l in range(mlp_layer):
        res = linear(res, mlp_hidden[l], scope='l'+str(l))
    return res
