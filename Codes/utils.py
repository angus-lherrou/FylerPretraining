#!/usr/bin/env python3

import torch

def pad_sequences(seqs, max_len=None):
  """Matrix of token ids padded with zeros"""

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in seqs)

  padded = torch.zeros(len(seqs), max_len, dtype=torch.long)

  for i, seq in enumerate(seqs):
    if len(seq) > max_len:
      seq = seq[:max_len]
    padded[i, :len(seq)] = torch.tensor(seq)

  return padded

if __name__ == "__main__":

  seqs = [[1, 2, 3, 4], [5, 6], [7]]
  output = pad_sequences(seqs, max_len=3)
  print(output)
