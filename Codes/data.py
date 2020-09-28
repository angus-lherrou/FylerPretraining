#!/usr/bin/env python3

import configparser, os, pandas, sys, pathlib, shutil, pickle
import tokenizer

model_dir = 'Model/'

alphabet_pickle = 'Model/alphabet.p'

diag_icd_file = 'DIAGNOSES_ICD.csv'
proc_icd_file = 'PROCEDURES_ICD.csv'
cpt_code_file = 'CPTEVENTS.csv'

class DatasetProvider:
  """Notes and ICD code data"""

  def __init__(self,
               input_dir,
               output_dir,
               cui_vocab_size,
               code_vocab_size):
    """Construct it"""

    self.input_dir = input_dir
    self.output_dir = output_dir

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # encounter ids -> icd code sets
    self.enc2codes = {}

    diag_code_path = os.path.join(self.output_dir, diag_icd_file)
    proc_code_path = os.path.join(self.output_dir, proc_icd_file)
    cpt_code_path = os.path.join(self.output_dir, cpt_code_file)

    self.index_codes(diag_code_path, 'ICD9_CODE', 'diag', 3)
    self.index_codes(proc_code_path, 'ICD9_CODE', 'proc', 2)
    self.index_codes(cpt_code_path, 'CPT_NUMBER', 'cpt', 5)

    # index inputs (cuis)
    self.input_tokenizer = tokenizer.Tokenizer(
      n_words=None if cui_vocab_size == 'all' else int(cui_vocab_size),
      lower=False,
      oov_token='oovtok')
    self.tokenize_input()

    # index outputs (codes)
    self.output_tokenizer = tokenizer.Tokenizer(
      n_words=None if code_vocab_size == 'all' else int(code_vocab_size),
      lower=False,
      oov_token='oovtok')
    self.tokenize_output()

  def index_codes(self, code_file, code_col, prefix, num_digits):
    """Map encounters to codes"""

    frame = pandas.read_csv(code_file, dtype='str')

    for id, code in zip(frame['HADM_ID'], frame[code_col]):
      if pandas.isnull(id):
        continue # some subjects skipped (e.g. 13567)
      if pandas.isnull(code):
        continue
      if id not in self.enc2codes:
        self.enc2codes[id] = set()

      short_code = '%s_%s' % (prefix, code[0:num_digits])
      self.enc2codes[id].add(short_code)

  def tokenize_input(self):
    """Read text and map tokens to ints"""

    x = [] # input documents
    for file_path in pathlib.Path(self.input_dir).glob('*.txt'):
      x.append(file_path.read_text())
    self.input_tokenizer.fit_on_texts(x)

    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.input_tokenizer, pickle_file)
    print('input vocab:', len(self.input_tokenizer.stoi))

  def tokenize_output(self):
    """Map codes to ints"""

    y = [] # prediction targets
    for _, codes in self.enc2codes.items():
      y.append(' '.join(codes))
    self.output_tokenizer.fit_on_texts(y)

    print('output vocab:', len(self.output_tokenizer.stoi))

  def load_as_sequences(self):
    """Make x and y"""

    x = []
    y = []

    # make a list of inputs and outputs to vectorize
    for file_path in pathlib.Path(self.input_dir).glob('*.txt'):
      if file_path.stem not in self.enc2codes:
        continue

      x.append(file_path.read_text())
      codes_as_string = ' '.join(self.enc2codes[file_path.stem])
      y.append(codes_as_string)

    # make x and y matrices
    x = self.input_tokenizer.texts_as_sets_to_seqs(x)
    y = self.output_tokenizer.texts_to_seqs(y)

    # column zero is empty
    # return x, y[:,1:]
    return x, y

if __name__ == "__main__":
  """Test dataset class"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'cui_vocab_size'),
    cfg.get('args', 'code_vocab_size'))

  inputs, outputs = dp.load()
  print(inputs[:2])
  print(outputs[:2])
