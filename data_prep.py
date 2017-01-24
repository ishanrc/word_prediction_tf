from itertools import islice
from collections import namedtuple
import tensorflow as tf
import numpy as np

LabeledData = namedtuple('LabeledData', ['inputs', 'labels'])

def window(seq, n):
  it = iter(seq)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for e in it:
    result = result[1:] + (e,)
    yield result


def read_vocab(sentence_file):
  vocab = set()
  with open(sentence_file) as f:
    for l in f:
      vocab.update(l.split())
  vocab = list(vocab)
  vocab.sort()
  return vocab

def get_one_hot(word, vocab):
  return [1 if word == v else 0 for v in vocab]
    
def get_n_grams(sentence_file, n=4):
  with open(sentence_file) as f:
    for l in f:
      l = l.split()
      yield from window(l, n)

def encode_data(sentence_file, vocab):
  inputs = []
  labels = []
  for g in getNGrams(sentence_file):
    inputs.append(getOneHot(g[0], vocab) + getOneHot(g[1], vocab) + getOneHot(g[2], vocab))
    labels.append(getOneHot(g[3], vocab))
  return LabeledData(inputs, labels)

def get_test_train(data, test_size):
    return LabeledData(data.inputs[:test_size], data.labels[:test_size]), LabeledData(data.inputs[test_size:], data.labels[test_size:])

def get_batch(data, i, batch_size):
  l = len(data.inputs)
  start = (batch_size * i) % l
  end = (batch_size * i + batch_size) % l
  if start < end:
    return data.inputs[start:end],data.labels[start:end]
  else:
    return data.inputs[start:] + data.inputs[:end], data.labels[start:] + data.labels[:end]
  
def build_nn(training_data, test_data):
  vocab_size = len(training_data.labels[0])
  x = tf.placeholder(tf.float32, shape=[None, vocab_size * 3])
  y_ = tf.placeholder(tf.float32, shape=[None, vocab_size])
  
  W = tf.Variable(tf.zeros([vocab_size * 3, vocab_size]))
  b = tf.Variable(tf.zeros([vocab_size]))
  y = tf.matmul(x, W) + b
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  # train
  for i in range(1000):
    batch_x, batch_y = get_batch(training_data, i, 100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={ x: test_data.inputs, y_: test_data.labels }))
