#!/usr/bin/env python3

import os

class DirDataReader:
  """Each label is in a subdirectory"""

  @staticmethod
  def read(path, label2int):
    """Subdirectories are class labels"""

    labels = []  # int labels
    samples = [] # examples as strings

    for label_dir in os.listdir(path):
      label_dir_path = os.path.join(path, label_dir)

      for file in os.listdir(label_dir_path):
        file_path = os.path.join(label_dir_path, file)
        file_text = open(file_path).read().rstrip()
        int_label = label2int[label_dir.lower()]
        samples.append(file_text)
        labels.append(int_label)

    return samples, labels

if __name__ == "__main__":

  samples, labels = DirDataReader.read(
    '/Users/Dima/Work/Data/Opioids1k/Train/',
    {'no':0, 'yes':1})

  print(samples[0][:100])
  print(labels[-25:])

  # some stats about the data
  lengths = [len(sample.split()) for sample in samples]
  print('average num of tokens:', round(sum(lengths)/len(lengths)))
  print('max_num_of_tokens:', max(lengths))
