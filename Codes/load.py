#!/usr/bin/env python3

import torch, configparser, os, sys, bow

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

  activations = {}
  def get_activation():
    def hook(model, input, output):
      print('model:', model)
      activations['relu_out'] = output.detach()

    return hook

  model.relu.register_forward_hook(get_activation())

  model(torch.randint(0, 100, (32, 132005)))
  print('activations:', activations['relu_out'])
  print('activations shape:', activations['relu_out'].shape)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
