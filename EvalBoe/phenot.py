#!/usr/bin/env python3

import numpy as np
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

# for sklearn grid search?
np.random.seed(1337)

import sys, os, pickle
sys.path.append('../Lib/')
sys.path.append('../Codes/')

import configparser, torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# my python modules
from dataphenot import DatasetProvider
import trans, utils, metrics

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def grid_search(x, y, scoring):
  """Find best model"""

  param_grid = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LogisticRegression(class_weight='balanced', max_iter=100000)

  gs = GridSearchCV(lr, param_grid, scoring=scoring, cv=10)
  gs.fit(x, y)
  print('best model:\n', str(gs.best_estimator_))

  return gs.best_estimator_

def run_evaluation_dense():
  """Use pre-trained patient representations"""

  x_train, y_train, x_test, y_test = data_dense()

  if cfg.get('data', 'classif_param') == 'search':
    classifier = grid_search(x_train, y_train, 'roc_auc')
  else:
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(x_train, y_train)

  probs = classifier.predict_proba(x_test)
  metrics.report_roc_auc(y_test, probs[:, 1])

def make_data_loader(input_seqs, batch_size, max_len):
  """Make DataLoader objects"""

  model_inputs = utils.pad_sequences(input_seqs, max_len=max_len)
  tensor_dataset = TensorDataset(model_inputs)
  data_loader = DataLoader(
    dataset=tensor_dataset,
    sampler=SequentialSampler(tensor_dataset),
    batch_size=batch_size)

  return data_loader

def get_dense_representations(model, x, batch_size=16):
  """Run sparse x through pretrain model and get dense representations"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  # todo: set max_len here to match training data?
  data_loader = make_data_loader(x, batch_size, None)

  # list of batched dense representations
  dense_x = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs = batch[0]

    with torch.no_grad():
      logits = model(batch_inputs, return_hidden=True)
      logits = logits.cpu()
      dense_x.append(logits.numpy())

  return np.vstack(dense_x)

def data_dense():
  """Data to feed into code prediction model"""

  train_data = os.path.join(base, cfg.get('data', 'train'))
  test_data = os.path.join(base, cfg.get('data', 'test'))

  # load model configuration
  pkl = open(cfg.get('data', 'config_pickle'), 'rb')
  config = pickle.load(pkl)

  # instantiate model and load parameters
  model = trans.TransformerEncoder(**config, save_config=False)
  state_dict = torch.load(cfg.get('data', 'model_file'))
  model.load_state_dict(state_dict)
  model.eval()

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    cfg.get('data', 'tokenizer_pickle'))

  x_train, y_train = train_data_provider.load_as_int_seqs()

  # make training vectors for target task
  x_train = get_dense_representations(model, x_train)

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    cfg.get('data', 'tokenizer_pickle'))

  x_test, y_test = test_data_provider.load_as_int_seqs()

  # make test vectors for target task
  x_test = get_dense_representations(model, x_test)

  return x_train, y_train, x_test, y_test

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  run_evaluation_dense()
