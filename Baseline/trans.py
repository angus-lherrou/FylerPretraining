#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

# from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformer import EncoderLayer

import os, configparser, random
import datareader, tokenizer, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)

class TransformerClassifier(nn.Module):
  """A transformative experience"""

  def __init__(self, num_classes=2 ):
    """We have some of the best constructors in the world"""

    super(TransformerClassifier, self).__init__()

    self.embed = nn.Embedding(
      num_embeddings=cfg.getint('data', 'vocab_size'),
      embedding_dim=cfg.getint('model', 'emb_dim'))

    trans_encoders = []
    for n in range(cfg.getint('model', 'n_layers')):
      trans_encoders.append(EncoderLayer(
        d_model=cfg.getint('model', 'emb_dim'),
        d_inner=cfg.getint('model', 'feedforw_dim'),
        n_head=cfg.getint('model', 'n_heads'),
        d_k=cfg.getint('model', 'emb_dim'),
        d_v=cfg.getint('model', 'emb_dim')))
    self.trans_encoders = nn.ModuleList(trans_encoders)

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.linear = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=num_classes)

    self.init_weights()

  def init_weights(self):
    """Initialize the weights"""

    self.embed.weight.data.uniform_(-0.1, 0.1)

  def forward(self, texts):
    """Moving forward"""

    output = self.embed(texts)

    # encoder input: (batch_size, seq_len, emb_dim)
    # encoder output: (batch_size, seq_len, emb_dim)
    for trans_encoder in self.trans_encoders:
      output, _ = trans_encoder(output)

    # average pooling
    output = torch.mean(output, dim=1)

    output = self.dropout(output)
    output = self.linear(output)

    return output

def make_data_loader(
  model_inputs,
  labels,
  batch_size,
  partition):
  """DataLoader objects for train or dev/test sets"""

  tensor_dataset = TensorDataset(
    model_inputs,
    torch.tensor(labels))

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

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  # scheduler = get_linear_schedule_with_warmup(
  #   optimizer,
  #   num_warmup_steps=100,
  #   num_training_steps=1000)

  best_roc_auc = 0
  optimal_epochs = 0

  for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_labels = batch
      optimizer.zero_grad()

      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      # scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss, roc_auc = evaluate(model, val_loader, weights)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f, val roc: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss, roc_auc))

    if roc_auc > best_roc_auc:
      best_roc_auc = roc_auc
      optimal_epochs = epoch

  return best_roc_auc, optimal_epochs

def evaluate(model, data_loader, weights):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  weights = weights.to(device)
  model.to(device)

  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_probs = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_labels)

    batch_logits = logits.detach().to('cpu')
    batch_labels = batch_labels.to('cpu')
    batch_probs = torch.nn.functional.softmax(batch_logits, dim=1)[:, 1]

    all_labels.extend(batch_labels.tolist())
    all_probs.extend(batch_probs.tolist())

    total_loss += loss.item()
    num_steps += 1

  av_loss = total_loss / num_steps
  roc_auc = roc_auc_score(all_labels, all_probs)

  return av_loss, roc_auc
 
def select_model():
  """Use validation set to tune"""

  tr_texts, tr_labels = datareader.DirDataReader.read(
    os.path.join(base, cfg.get('data', 'train')),
    {'no':0, 'yes':1})

  tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    tr_texts, tr_labels, test_size=0.20, random_state=2020)

  tok = tokenizer.Tokenizer(cfg.getint('data', 'vocab_size'))
  tok.fit_on_texts(tr_texts)

  tr_texts = tok.texts_as_sets_to_seqs(tr_texts)
  val_texts = tok.texts_as_sets_to_seqs(val_texts)

  # todo: what's up with max length?

  train_loader = make_data_loader(
    utils.pad_sequences(tr_texts),
    tr_labels,
    cfg.getint('model', 'batch_size'),
    'train')
  val_loader = make_data_loader(
    utils.pad_sequences(val_texts),
    val_labels,
    cfg.getint('model', 'batch_size'),
    'dev')
  print('loaded %d training and %d validation samples' % \
        (len(tr_texts), len(val_texts)))

  model = TransformerClassifier()

  label_counts = torch.bincount(torch.tensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  best_roc, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    weights,
    cfg.getint('model', 'n_epochs'))
  print('roc auc %.3f after %d epochs' % (best_roc, optimal_epochs))

  return optimal_epochs

def run_evaluation(optimal_epochs):
  """Train a model and evaluate on the test set"""

  tr_texts, tr_labels = datareader.DirDataReader.read(
    os.path.join(base, cfg.get('data', 'train')),
    {'no':0, 'yes':1})

  test_texts, test_labels = datareader.DirDataReader.read(
    os.path.join(base, cfg.get('data', 'test')),
    {'no':0, 'yes':1})

  tok = tokenizer.Tokenizer(cfg.getint('data', 'vocab_size'))
  tok.fit_on_texts(tr_texts)

  tr_texts = tok.texts_as_sets_to_seqs(tr_texts)
  test_texts = tok.texts_as_sets_to_seqs(test_texts)

  train_loader = make_data_loader(
    utils.pad_sequences(tr_texts),
    tr_labels,
    cfg.getint('model', 'batch_size'),
    'train')
  test_loader = make_data_loader(
    utils.pad_sequences(test_texts),
    test_labels,
    cfg.getint('model', 'batch_size'),
    'test')
  print('loaded %d training and %d test samples' % \
        (len(tr_texts), len(test_texts)))

  model = TransformerClassifier()

  label_counts = torch.bincount(torch.tensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  fit(model, train_loader, test_loader, weights, optimal_epochs)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  optimal_epochs = select_model()
  run_evaluation(optimal_epochs)
