#!/usr/bin/env python3

from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

import sys, os, pickle
sys.path.append('../Lib/')
sys.path.append('../Codes/')

from sklearn.model_selection import train_test_split

import configparser, torch, shutil

# my python modules
from dataphenot import DatasetProvider
import bow, utils, metrics

# local model path
model_path = 'Model/model.pt'
model_dir = 'Model/'

def make_data_loader(model_inputs, model_outputs, batch_size, partition):
  """DataLoader objects for train or dev/test sets"""

  tensor_dataset = TensorDataset(model_inputs, model_outputs)

  # use sequential sampler for dev and test
  if partition == 'train':
    sampler = RandomSampler(tensor_dataset)
  else:
    sampler = SequentialSampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=sampler,
    batch_size=batch_size)

  return data_loader

def fit(model, train_loader, val_loader, weights, n_epochs):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  weights = weights.to(device)
  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  best_loss = float('inf')
  optimal_epochs = 0

  for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      optimizer.zero_grad()

      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_outputs = batch

      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_outputs)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_tr_loss = train_loss / num_train_steps
    val_loss = evaluate(model, val_loader, weights)
    print('ep: %d, steps: %d, tr loss: %.4f, val loss: %.4f' % \
          (epoch, num_train_steps, av_tr_loss, val_loss))

    if val_loss < best_loss:
      print('loss improved, saving model...')
      torch.save(model.state_dict(), model_path)
      best_loss = val_loss
      optimal_epochs = epoch

  return best_loss, optimal_epochs

def evaluate(model, data_loader, weights):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)
  total_loss, num_steps = 0, 0

  model.eval()

  for batch in data_loader:

    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_outputs = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_outputs)

    total_loss += loss.item()
    num_steps += 1

  av_loss = total_loss / num_steps
  return av_loss

def main():
  """Data to feed into code prediction model"""

  # load model configuration
  pkl = open(cfg.get('data', 'config_pickle'), 'rb')
  config = pickle.load(pkl)

  # instantiate model and load parameters
  model = bow.BagOfWords(**config, save_config=False)
  state_dict = torch.load(cfg.get('data', 'model_file'))
  model.load_state_dict(state_dict)
  model.eval()

  # replace the old classification layer
  model.classifier = torch.nn.Linear(
    in_features=config['hidden_units'],
    out_features=2)

  # load training data first
  train_data_provider = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'tokenizer_pickle'))

  x_train, y_train = train_data_provider.load_as_int_seqs()

  x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.10, random_state=2020)

  x_train = utils.sequences_to_matrix(
    x_train,
    config['input_vocab_size'])
  x_val = utils.sequences_to_matrix(
    x_val,
    config['input_vocab_size'])

  train_loader = make_data_loader(
    x_train,
    torch.tensor(y_train),
    cfg.getint('model', 'batch'),
    'train')
  val_loader = make_data_loader(
    x_val,
    torch.tensor(y_val),
    cfg.getint('model', 'batch'),
    'dev')

  label_counts = torch.bincount(torch.tensor(y_train))
  weights = len(y_train) / (2.0 * label_counts)

  best_loss, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    weights,
    cfg.getint('model', 'epochs'))
  print('best loss %.4f after %d epochs' % (best_loss, optimal_epochs))

  # now load the test set
  test_data_provider = DatasetProvider(
    os.path.join(base, cfg.get('data', 'test')),
    cfg.get('data', 'tokenizer_pickle'))

  x_test, y_test = test_data_provider.load_as_int_seqs()
  x_test = utils.sequences_to_matrix(
    x_test,
    config['input_vocab_size'])

  return x_train, y_train, x_test, y_test

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  os.mkdir(model_dir)

  main()
