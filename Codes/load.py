#!/usr/bin/env python3

import torch, configparser, os, sys, bow

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
  """My main main"""

  model = bow.BagOfEmbeddings(
    input_vocab_size=cfg.getint('args', 'cui_vocab_size'),
    output_vocab_size=cfg.getint('args', 'code_vocab_size'),
    embed_dim=cfg.getint('model', 'embed'),
    hidden_units=cfg.getint('model', 'hidden'),
    dropout_rate=cfg.getfloat('model', 'dropout'))

  state_dict = torch.load('Model/model.pt')
  model.load_state_dict(state_dict)
  model.eval()
  print(model)

  model.relu.register_forward_hook(get_activation('relu'))

  input = torch.randint(0, 100, (32, 132005))
  output = model(input)

  print(output.shape)
  print(activation['relu'])

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
