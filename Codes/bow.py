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

  def __init__(self, input_vocab_size, output_vocab_size):
    """Constructor"""

    super(BagOfEmbeddings, self).__init__()

    self.embed = nn.Embedding(
      num_embeddings=input_vocab_size,
      embedding_dim=cfg.getint('model', 'embed'))

    self.hidden = nn.Linear(
      in_features=cfg.getint('model', 'embed'),
      out_features=cfg.getint('model', 'hidden'))

    self.relu = nn.ReLU()

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.classifier = nn.Linear(
      in_features=cfg.getint('model', 'hidden'),
      out_features=output_vocab_size)

  def forward(self, texts):
    """Forward pass"""

    output = self.embed(texts)
    output = torch.mean(output, dim=1)
    output = self.hidden(output)
    output = self.relu(output)
    output = self.dropout(output)
    output = self.classifier(output)

    return output

def make_data_loader(input_seqs, model_outputs, batch_size, partition):
  """DataLoader objects for train or dev/test sets"""

  model_inputs = utils.pad_sequences(input_seqs, max_len=None)

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

  best_metric = -1
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
    val_loss, metric = evaluate(model, val_loader)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f, val roc: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss, metric))

    if metric > best_metric:
      best_metric = metric
      optimal_epochs = epoch

  return best_metric, optimal_epochs

def evaluate(model, data_loader):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = nn.BCEWithLogitsLoss()
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = criterion(logits, batch_labels)

    batch_logits = logits.detach().to('cpu')
    batch_labels = batch_labels.to('cpu')
    # batch_preds = np.argmax(batch_logits, axis=1)
    batch_predictions = torch.argmax(batch_logits, dim=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_predictions.tolist())

    total_loss += loss.item()
    num_steps += 1

  av_loss = total_loss / num_steps
  f1 = f1_score(all_labels, all_predictions, average='micro')

  return av_loss, f1
 
def main():
  """Fine-tune bert"""

  dp = data.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'cui_vocab_size'),
    cfg.get('args', 'code_vocab_size'))
  in_seqs, out_seqs = dp.load()

  tr_in_seqs, val_in_seqs, tr_out_seqs, val_out_seqs = train_test_split(
    in_seqs, out_seqs, test_size=0.15, random_state=2020)

  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(val_in_seqs)))

  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  print('longest cui sequence:', max_cui_seq_len)

  max_code_seq_len = max(len(seq) for seq in tr_out_seqs)
  print('longest code sequence:', max_code_seq_len)

  train_loader = make_data_loader(
    tr_in_seqs,
    utils.sequences_to_matrix(tr_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'train')

  val_loader = make_data_loader(
    val_in_seqs,
    utils.sequences_to_matrix(val_out_seqs, len(dp.output_tokenizer.stoi)),
    cfg.getint('model', 'batch'),
    'dev')

  model = BagOfEmbeddings(
    input_vocab_size=len(dp.input_tokenizer.stoi),
    output_vocab_size=len(dp.output_tokenizer.stoi))

  best_f1, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    cfg.getint('model', 'epochs'))
  print('roc auc %.3f after %d epochs' % (best_f1, optimal_epochs))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
