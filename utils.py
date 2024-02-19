# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for offline RL."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import re
import csv
import collections

import numpy as np
import tensorflow.compat.v1 as tf
from dataset import Transition

def read_csv_file(file_path, num_obs, num_act):
  rows = []
  with open(file_path, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
      rows.append(row)
  rows = np.array(rows, dtype=np.float32)
  state_num = num_obs
  action_num = num_act
  rows[:, -1] = 1 - rows[:, -1]
  transaction = Transition(s1=rows[0:-1, 0:state_num],
                           a1=rows[0:-1, state_num:state_num+action_num],
                           s2=rows[1:, 0:state_num],
                           a2=rows[1:, state_num:state_num+action_num],
                           reward=rows[0:-1, -2],
                           discount=rows[0:-1, -1])
  return transaction

def merge_transitions(trans1, trans2):
  
  cs1 = np.concatenate([trans1.s1, trans2.s1])
  cs2 = np.concatenate([trans1.s2, trans2.s2])
  ca1 = np.concatenate([trans1.a1, trans2.a1])
  ca2 = np.concatenate([trans1.a2, trans2.a2])
  creward = np.concatenate([trans1.reward, trans2.reward])
  cdiscount = np.concatenate([trans1.discount, trans2.discount])
  transaction = Transition(s1=cs1,
                           a1=ca1,
                           s2=cs2,
                           a2=ca2,
                           reward=creward,
                           discount=cdiscount)
  return transaction

def numpy2tensor(tran):
  transaction = Transition(s1=tf.convert_to_tensor(tran.s1, tf.float32),
                           a1=tf.convert_to_tensor(tran.a1, tf.float32),
                           s2=tf.convert_to_tensor(tran.s2, tf.float32),
                           a2=tf.convert_to_tensor(tran.a2, tf.float32),
                           reward=tf.convert_to_tensor(tran.reward, tf.float32),
                           discount=tf.convert_to_tensor(tran.discount, tf.float32))
  return transaction

def get_summary_str(step=None, info=None, prefix=''):
  summary_str = prefix
  if step is not None:
    summary_str += 'Step %d; ' % (step)
  for key, val in info.items():
    if isinstance(val, (int, np.int32, np.int64)):
      summary_str += '%s %d; ' % (key, val)
    elif isinstance(val, (float, np.float32, np.float64)):
      summary_str += '%s %.4g; ' % (key, val)
  return summary_str


def write_summary(summary_writer, step, info):
  with summary_writer.as_default():
    for key, val in info.items():
      if isinstance(
          val, (int, float, np.int32, np.int64, np.float32, np.float64)):
        tf.compat.v2.summary.scalar(name=key, data=val, step=step)


def soft_variables_update(source_variables, target_variables, tau=1.0):
  for (v_s, v_t) in zip(source_variables, target_variables):
    v_t.assign((1 - tau) * v_t + tau * v_s)


def shuffle_indices_with_steps(n, steps=1, rand=None):
  """Randomly shuffling indices while keeping segments."""
  if steps == 0:
    return np.arange(n)
  if rand is None:
    rand = np.random
  n_segments = int(n // steps)
  n_effective = n_segments * steps
  batch_indices = rand.permutation(n_segments)
  batches = np.arange(n_effective).reshape([n_segments, steps])
  shuffled_batches = batches[batch_indices]
  shuffled_indices = np.arange(n)
  shuffled_indices[:n_effective] = shuffled_batches.reshape([-1])
  return shuffled_indices


def clip_by_eps(x, spec, eps=0.0):
  return tf.clip_by_value(
      x, spec.minimum + eps, spec.maximum - eps)


def get_optimizer(name):
  """Get an optimizer generator that returns an optimizer according to lr."""
  if name == 'adam':
    def adam_opt_(lr):
      return tf.keras.optimizers.Adam(lr=lr)
    return adam_opt_
  else:
    raise ValueError('Unknown optimizer %s.' % name)


def load_variable_from_ckpt(ckpt_name, var_name):
  var_name_ = '/'.join(var_name.split('.')) + '/.ATTRIBUTES/VARIABLE_VALUE'
  return tf.train.load_variable(ckpt_name, var_name_)


def soft_relu(x):
  """Compute log(1 + exp(x))."""
  # Note: log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
  #       log(1 - sigmoid(x)) = - soft_relu(x)
  return tf.log(1.0 + tf.exp(-tf.abs(x))) + tf.maximum(x, 0.0)


@tf.custom_gradient
def relu_v2(x):
  """Relu with modified gradient behavior."""
  value = tf.nn.relu(x)
  def grad(dy):
    if_y_pos = tf.cast(tf.greater(dy, 0.0), tf.float32)
    if_x_pos = tf.cast(tf.greater(x, 0.0), tf.float32)
    return (if_y_pos * if_x_pos + (1.0 - if_y_pos)) * dy
  return value, grad


@tf.custom_gradient
def clip_v2(x, low, high):
  """Clipping with modified gradient behavior."""
  value = tf.minimum(tf.maximum(x, low), high)
  def grad(dy):
    if_y_pos = tf.cast(tf.greater(dy, 0.0), tf.float32)
    if_x_g_low = tf.cast(tf.greater(x, low), tf.float32)
    if_x_l_high = tf.cast(tf.less(x, high), tf.float32)
    return (if_y_pos * if_x_g_low +
            (1.0 - if_y_pos) * if_x_l_high) * dy
  return value, grad


class Flags(object):

  def __init__(self, **kwargs):
    for key, val in kwargs.items():
      setattr(self, key, val)


def get_datetime():
  now = datetime.datetime.now().isoformat()
  now = re.sub(r'\D', '', now)[:-6]
  return now


def maybe_makedirs(log_dir):
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)
