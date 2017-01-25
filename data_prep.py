from itertools import islice
from collections import namedtuple

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
  for g in get_n_grams(sentence_file):
    inputs.append(get_one_hot(g[0], vocab) + get_one_hot(g[1], vocab) + get_one_hot(g[2], vocab))
    labels.append(get_one_hot(g[3], vocab))
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
  
