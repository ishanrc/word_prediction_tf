def readVocab(sentences_file):
  vocab = set()
  with open(sentences_file) as f:
    for l in f:
      vocab.update(l.split())
  vocab = list(vocab)
  vocat.sort()
  return vocab

def getOneHot(word, vocab):
  return [1 if word == v else 0 for v in vocab]
    
def encodeData(sentences_file, vocab):
  data = []
  with open(sentences_file) as f:
    for l in f:
      l = l.split()
      data.append(sum(map(lambda w: getOneHot(w, vocab), l), []))
  return data
