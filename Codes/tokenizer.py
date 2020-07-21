#!/usr/bin/env python3

import collections

class Tokenizer:
  """Tokenization and vectorization"""

  def __init__(self, n_words, lower=False, oov_token='oovtok'):
    """Construction deconstruction"""

    self.stoi = {}
    self.itos = {}

    self.n_words = None if n_words is None else n_words-1
    self.lower = lower
    self.oov = oov_token

    self.stoi[oov_token] = 0
    self.itos[0] = oov_token

  def fit_on_texts(self, texts):
    """Fit on a list of document strings"""

    counts = collections.Counter()

    for text in texts:
      tokens = text.split()
      counts.update(tokens)

    index = 1
    for token, _ in counts.most_common(self.n_words):
      self.stoi[token] = index
      self.itos[index] = token
      index += 1

  def texts_to_seqs(self, texts):
    """List of strings to list of int sequences"""

    sequences = []
    for text in texts:

      sequence = []
      for token in text.split():
        if token in self.stoi:
          sequence.append(self.stoi[token])
        else:
          sequence.append(self.stoi[self.oov])

      sequences.append(sequence)

    return sequences

  def texts_as_sets_to_seqs(self, texts):
    """Same as texts_to_sequences but treat texts as sets"""

    sequences = []
    for text in texts:

      sequence = []
      for token in set(text.split()):
        if token in self.stoi:
          sequence.append(self.stoi[token])
        else:
          sequence.append(self.stoi[self.oov])

      sequences.append(sequence)

    return sequences

if __name__ == "__main__":

  sents = ['it is happening again',
           'the owls are not what they seem',
           'again and again',
           'the owls are happening']

  tokenizer = Tokenizer(n_words=5)

  tokenizer.fit_on_texts(sents)
  print(tokenizer.stoi)
  print(tokenizer.itos)

  seqs = tokenizer.texts_to_sequences(sents)
  print(seqs)