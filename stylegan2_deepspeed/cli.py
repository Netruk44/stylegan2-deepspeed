import argparse
import deepspeed
from stylegan2_deepspeed.trainer import Trainer

def parse_arguments(argument_override = None):
  parser = argparse.ArgumentParser(description='StyleGAN2-deepspeed')

  parser.add_argument('-d', '--data_dir', default='./data', type=str, help='data directory containing training images')
  parser.add_argument('-r', '--results_dir', default='./results', type=str, help='directory to store example images created during training')
  parser.add_argument('-m', '--models_dir', default='./models', type=str, help='directory to store model checkpoints')
  parser.add_argument('-n', '--name', default='default', type=str, help='name of the model to train')
  parser.add_argument('--new', default=False, type=bool, help='start training over for a model')
  parser.add_argument('-s', '--image_size', default=128, type=int, help='size of image to generate, must be a power of 2 (32, 64, 128, 256, 512, ...)')
  parser.add_argument('--network_capacity', default=16, type=int, help='network capacity (default 16), lower values use less memory but produce lower quality results')
  parser.add_argument('--learning_rate', default=2e-4, type=float, help='the learning rate')
  parser.add_argument('--save_every', default=1000, type=int, help='how often to checkpoint')
  parser.add_argument('-e', '--ema_beta', default=0.99, type=float, help='beta value for the exponential moving average (usually between 0.99 and 0.9999)')
  parser.add_argument('--ema_k', default=5, type=int, help='how often to update the ema')
  parser.add_argument('--latent_dim', default=512, type=int, help='the size of the latent dimension')
  parser.add_argument('--lookahead', default=True, type=bool, help='use lookahead with optimizer')
  parser.add_argument('--lookahead_alpha', default=0.5, type=float, help='alpha parameter for lookahead implementation')
  parser.add_argument('--lookahead_k', default=5, type=int, help='k parameter for lookahead implementation')
  parser.add_argument('-c', '--checkpoint_every', default=1000, type=int, help='number of iterations between automatic checkpoints')
  parser.add_argument('--gen_load_from', default='', type=str, help='tag to load generator checkpoint from')
  parser.add_argument('--disc_load_from', default='', type=str, help='tag to load discriminator checkpoint from')

  parser.add_argument('--local_rank', default=0, type=int, help='the rank of this process')

  parser = deepspeed.add_config_arguments(parser)

  if argument_override is not None:
    return parser.parse_args(argument_override)
  else:
    return parser.parse_args()


def main():
  deepspeed.init_distributed()
  args = parse_arguments()
  Trainer().train(args)