#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

from transformer import EncoderLayer

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

class TransformerEncoder(nn.Module):

  def __init__(
    self,
    input_vocab_size,
    output_vocab_size,
    max_len,
    save_config=True):
    """Constructor"""

    super(TransformerEncoder, self).__init__()

    self.embed = nn.Embedding(
      num_embeddings=input_vocab_size,
      embedding_dim=cfg.getint('model', 'd_model'))

    trans_encoders = []
    for n in range(cfg.getint('model', 'n_layers')):
      trans_encoders.append(EncoderLayer(
        d_model=cfg.getint('model', 'd_model'),
        d_inner=cfg.getint('model', 'd_inner'),
        n_head=cfg.getint('model', 'n_head'),
        d_k=cfg.getint('model', 'd_model'),
        d_v=cfg.getint('model', 'd_model')))
    self.trans_encoders = nn.ModuleList(trans_encoders)

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.classifier = nn.Linear(
      in_features=cfg.getint('model', 'd_model'),
      out_features=output_vocab_size)

    # save configuration for loading later
    if save_config:
      config = dict(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        d_model=cfg.getint('model', 'd_model'),
        d_inner=cfg.getint('model', 'd_inner'),
        n_head=cfg.getint('model', 'n_head'),
        d_k=cfg.getint('model', 'd_k'),
        d_v=cfg.getint('model', 'd_v'),
        dropout_rate=cfg.getfloat('model', 'dropout'),
        max_len=max_len)

      pickle_file = open(config_path, 'wb')
      pickle.dump(config, pickle_file)

    self.init_weights()

  def init_weights(self):
    """Initialize the weights"""

    self.embed.weight.data.uniform_(-0.1, 0.1)

  def forward(self, texts, return_features=False):
    """Optionally return hidden layer activations"""

    output = self.embed(texts)

    # encoder input: (batch_size, seq_len, emb_dim)
    # encoder output: (batch_size, seq_len, emb_dim)
    for trans_encoder in self.trans_encoders:
      output, _ = trans_encoder(output)

    # use the first (CLS) token
    features = output[:, 0, :]

    output = self.dropout(features)
    output = self.classifier(output)

    if return_features:
      return features
    else:
      return output

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

def fit(model, train_loader, val_loader, n_epochs):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = nn.BCEWithLogitsLoss()

  optimizer = torch.optim.AdamW(
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
    in_seqs, out_seqs, test_size=0.20, random_state=2020)
  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(val_in_seqs)))

  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  max_code_seq_len = max(len(seq) for seq in tr_out_seqs)
  print('longest cui sequence:', max_cui_seq_len)
  print('longest code sequence:', max_code_seq_len)

  train_loader = make_data_loader(
    utils.pad_sequences(tr_in_seqs, max_len=cfg.getint('args', 'max_len')),
    utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'train')

  val_loader = make_data_loader(
    utils.pad_sequences(val_in_seqs, max_len=cfg.getint('args', 'max_len')),
    utils.sequences_to_matrix(val_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'dev')

  model = TransformerEncoder(
    input_vocab_size=len(dp.input_tokenizer.stoi),
    output_vocab_size=len(dp.output_tokenizer.stoi),
    max_len=cfg.getint('args', 'max_len'))

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
