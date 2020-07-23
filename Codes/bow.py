#!/usr/bin/env python3

import sys

from sklearn.metrics import f1_score

sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

import os, configparser, random
import data, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)

class BagOfEmbeddings(nn.Module):

  def __init__(self, num_class=3000):
    """Constructor"""

    super(BagOfEmbeddings, self).__init__()

    self.embed = nn.Embedding(
      num_embeddings=max_cuis,
      embedding_dim=cfg.getint('model', 'emb_dim'))

    self.hidden = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=cfg.getint('model', 'hidden_size'))

    self.relu = nn.ReLU()

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.classifier = nn.Linear(
      in_features=cfg.getint('model', 'hidden_size'),
      out_features=num_class)

  def forward(self, texts):
    """Forward pass"""

    output = self.embed(texts)
    output = torch.mean(output, dim=1)
    output = self.hidden(output)
    output = self.relu(output)
    output = self.dropout(output)
    output = self.classifier(output)

    return output

def make_data_loader(input_seqs, output_seqs, batch_size, max_len, partition):
  """DataLoader objects for train or dev/test sets"""

  model_inputs = utils.pad_sequences(input_seqs, max_len)
  model_outputs = utils.pad_sequences(output_seqs, max_len=None)

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

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000)

  best_roc_auc = -1
  optimal_epochs = -1

  for epoch in range(1, n_epochs + 1):

    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:

      optimizer.zero_grad()

      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_labels = batch

      logits = model(batch_inputs)
      loss = criterion(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss, roc_auc = evaluate(model, val_loader)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f, val roc: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss, roc_auc))

    if roc_auc > best_roc_auc:
      best_roc_auc = roc_auc
      optimal_epochs = epoch

  return best_roc_auc, optimal_epochs

def evaluate(model, data_loader):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = torch.nn.CrossEntropyLoss()
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_probs = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = criterion(logits, batch_labels)

    batch_logits = logits.detach().to('cpu')
    batch_labels = batch_labels.to('cpu')
    batch_probs = torch.nn.functional.softmax(batch_logits, dim=1)[:, 1]

    all_labels.extend(batch_labels.tolist())
    all_probs.extend(batch_probs.tolist())

    total_loss += loss.item()
    num_steps += 1

  av_loss = total_loss / num_steps
  f1 = f1_score(all_labels, all_probs, average='micro')

  return av_loss, f1
 
def main():
  """Fine-tune bert"""

  dp = data.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'max_cuis'),
    cfg.get('args', 'max_codes'))
  inputs, outputs = dp.load()

  tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    inputs, outputs, test_size=0.20, random_state=2020)

  train_loader = make_data_loader(
    tr_texts,
    tr_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'train')

  val_loader = make_data_loader(
    val_texts,
    val_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'dev')

  print('loaded %d training and %d validation samples' % \
        (len(tr_texts), len(val_texts)))

  model = BagOfEmbeddings()

  best_roc, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    cfg.getint('model', 'num_epochs'))
  print('roc auc %.3f after %d epochs' % (best_roc, optimal_epochs))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  max_cuis = cfg.get('args', 'max_cuis')
  max_cuis = None if max_cuis == 'all' else int(max_cuis)

  main()
