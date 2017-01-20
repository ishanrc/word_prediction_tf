from itertools import islice
from collections import namedtuple

def window(seq, n):
  it = iter(seq)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for e in it:
    result = result[1:] + (e,)
    yield result


def readVocab(sentence_file):
  vocab = set()
  with open(sentence_file) as f:
    for l in f:
      vocab.update(l.split())
  vocab = list(vocab)
  vocab.sort()
  return vocab

def getOneHot(word, vocab):
  return [1 if word == v else 0 for v in vocab]
    
def getNGrams(sentence_file, n=4):
  with open(sentence_file) as f:
    for l in f:
      l = l.split()
      yield from window(l, n)

def encodeData(sentence_file, vocab):
  inputs = []
  labels = []
  LabeledData = namedtuple('LabeledData', ['inputs', 'labels'])
  for g in getNGrams(sentence_file):
    inputs.append(getOneHot(g[0], vocab) + getOneHot(g[1], vocab) + getOneHot(g[2], vocab))
    labels.append(getOneHot(g[3], vocab))
  return LabeledData(inputs, labels)
