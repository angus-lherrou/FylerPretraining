#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

import os, configparser, random, pickle
import data, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)

# model and model config locations
model_path = 'Model/model.pt'
config_path = 'Model/config.p'

class BagOfWords(nn.Module):

  def __init__(
    self,
    input_vocab_size,
    output_vocab_size,
    hidden_units1,
    hidden_units2,
    dropout_rate,
    save_config=True):
    """Constructor"""

    super(BagOfWords, self).__init__()

    self.hidden1 = nn.Linear(
      in_features=input_vocab_size,
      out_features=hidden_units1)
    self.activation1 = nn.ReLU()

    self.hidden2 = nn.Linear(
      in_features=hidden_units1,
      out_features=hidden_units2)
    self.activation2 = nn.ReLU()

    self.dropout = nn.Dropout(dropout_rate)
    self.classifier = nn.Linear(
      in_features=hidden_units2,
      out_features=output_vocab_size)

    # save configuration for loading later
    if save_config:
      config = {
        'input_vocab_size': input_vocab_size,
        'output_vocab_size': output_vocab_size,
        'hidden_units1': hidden_units1,
        'hidden_units2': hidden_units2,
        'dropout_rate': dropout_rate}
      pickle_file = open(config_path, 'wb')
      pickle.dump(config, pickle_file)

  def forward(self, texts, return_hidden=False):
    """Optionally return hidden layer activations"""

    output1 = self.hidden1(texts)
    output1 = self.activation1(output1)

    output2 = self.hidden2(output1)
    output2 = self.activation2(output2)

    # residual connection
    output = output1 + output2

    output = self.dropout(output)
    output = self.classifier(output)

    if return_hidden:
      return output2
    else:
      return output

def make_data_loader(model_inputs, model_outputs, batch_size, partition):
  """DataLoader objects for train or dev/test sets"""

  # e.g. transformers take input ids and attn masks
  if type(model_inputs) is tuple:
    tensor_dataset = TensorDataset(*model_inputs, model_outputs)
  else:
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

def fit(model, train_loader, val_loader, n_epochs):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = nn.BCEWithLogitsLoss()

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
      loss = criterion(logits, batch_outputs)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_tr_loss = train_loss / num_train_steps
    val_loss = evaluate(model, val_loader)
    print('ep: %d, steps: %d, tr loss: %.4f, val loss: %.4f' % \
          (epoch, num_train_steps, av_tr_loss, val_loss))

    if val_loss < best_loss:
      print('loss improved, saving model...')
      torch.save(model.state_dict(), model_path)
      best_loss = val_loss
      optimal_epochs = epoch

  return best_loss, optimal_epochs

def evaluate(model, data_loader):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = nn.BCEWithLogitsLoss()
  total_loss, num_steps = 0, 0

  model.eval()

  for batch in data_loader:

    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_outputs = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = criterion(logits, batch_outputs)

    total_loss += loss.item()
    num_steps += 1

  av_loss = total_loss / num_steps
  return av_loss
 
def main():
  """My main main"""

  dp = data.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'cui_vocab_size'),
    cfg.get('args', 'code_vocab_size'))
  in_seqs, out_seqs = dp.load_as_sequences()

  tr_in_seqs, val_in_seqs, tr_out_seqs, val_out_seqs = train_test_split(
    in_seqs, out_seqs, test_size=0.10, random_state=2020)

  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(val_in_seqs)))

  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  print('longest cui sequence:', max_cui_seq_len)

  max_code_seq_len = max(len(seq) for seq in tr_out_seqs)
  print('longest code sequence:', max_code_seq_len)

  train_loader = make_data_loader(
    utils.sequences_to_matrix(tr_in_seqs, len(dp.input_tokenizer.stoi)),
    utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'train')

  val_loader = make_data_loader(
    utils.sequences_to_matrix(val_in_seqs, len(dp.input_tokenizer.stoi)),
    utils.sequences_to_matrix(val_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'dev')

  model = BagOfWords(
    input_vocab_size=len(dp.input_tokenizer.stoi),
    output_vocab_size=len(dp.output_tokenizer.stoi),
    hidden_units1=cfg.getint('model', 'hidden'),
    hidden_units2=cfg.getint('model', 'hidden'),
    dropout_rate=cfg.getfloat('model', 'dropout'))

  best_loss, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    cfg.getint('model', 'epochs'))
  print('best loss %.4f after %d epochs' % (best_loss, optimal_epochs))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
