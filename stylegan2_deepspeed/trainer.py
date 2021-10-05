import deepspeed
from math import log2
import multiprocessing
from random import random
from stylegan2_deepspeed.dataset import Dataset, cycle
from stylegan2_deepspeed.ema import EMA
from stylegan2_deepspeed.lookahead import Lookahead
import stylegan2_deepspeed.stylegan2 as stylegan2
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

NUM_CORES = multiprocessing.cpu_count()

def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def forever():
  while True:
    yield

def create_generator(args, device):
  return stylegan2.StylizedGenerator(args.image_size, network_capacity=args.network_capacity).cuda(device)

class TrainingRun():
  def __init__(self, gen, gen_opt, disc, disc_opt, args, loader, batch_size, device):
    self.gen = gen
    self.gen_opt = gen_opt
    self.disc = disc
    self.disc_opt = disc_opt
    self.loader = loader
    self.batch_size = batch_size
    self.device = device

    self.image_size = args.image_size
    self.num_layers = int(log2(self.image_size) - 1)
    self.latent_dim = args.latent_dim
    self.lookahead = args.lookahead
    self.lookahead_k = args.lookahead_k

    self.rank = args.local_rank
    self.is_primary = self.rank == 0

    if self.is_primary:
      self.gen_ema = create_generator(args, device)
      self.gen_ema.requires_grad_(False)

      self.ema_k = args.ema_k
      self.ema = EMA(args.ema_beta)

    self.total_steps = 0
    self.mixed_prob = 0.9
  
  def get_image_batch(self):
    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
    style = get_latents_fn(self.batch_size, self.num_layers, self.latent_dim, self.device)
    noise = image_noise(self.batch_size, self.image_size, self.device)

    return self.gen.forward(style, noise)
  
  def step(self):
    # Train Discriminator
    generated_images = self.get_image_batch()
    fake_output_loss, fake_q_loss = self.disc.forward(generated_images)

    image_batch = next(self.loader).cuda(self.device)
    image_batch.requires_grad_()
    real_output_loss, real_q_loss = self.disc.forward(image_batch)

    disc_loss = self.D_loss_fn(real_output_loss, fake_output_loss)
    self.disc.backward(disc_loss)
    self.disc.step()

    # Train Generator
    generated_images = self.get_image_batch()
    fake_output_loss, _ = self.disc.forward(generated_images)
    real_output_loss = None

    gen_loss = self.G_loss_fn(fake_output_loss, real_output_loss)
    self.gen.backward(gen_loss)
    self.gen.step()

    # Joint lookahead update
    if self.lookahead and (self.total_steps + 1) % self.lookahead_k == 0:
      self.gen_opt.lookahead_step()
      self.disc_opt.lookahead_step()

    # EMA update
    if self.is_primary and (self.total_steps + 1) % self.ema_k == 0:
      self.ema.update_ema(self.gen, self.gen_ema)

    self.total_steps = self.total_steps + 1
  
  def train(self):
    for _ in tqdm(forever()):
      # TODO: Check-ins (image dump & print to console)
      # TODO: Save checkpoints
      self.step()
    

class Trainer():
  def __init__(self, results_directory = './results', models_directory = './models'):
    self.results_directory = results_directory
    self.models_directory = models_directory

    self.D_loss_fn = hinge_loss
    self.G_loss_fn = gen_hinge_loss
  
  def train(
    self,
    args,
  ):
    name = args.name
    data_directory = args.data_dir
    save_every = args.save_every
    image_size = args.image_size
    network_capacity = args.network_capacity
    learning_rate = args.learning_rate
    ema_beta = args.ema_beta
    latent_dim = args.latent_dim

    world_size = torch.distributed.get_world_size()
    is_ddp = world_size > 1
    rank = args.local_rank

    ttur_mult = 2.
    mixed_prob = 0.9

    # TODO: Load from checkpoint if it exists

    gen = create_generator(args, rank)
    disc = stylegan2.AugmentedDiscriminator(image_size, network_capacity=network_capacity).cuda(rank)

    gen_opt = Adam(gen.parameters(), lr = learning_rate, betas=(0.5, 0.9))
    disc_opt = Adam(disc.parameters(), lr = learning_rate * ttur_mult, betas=(0.5, 0.9))

    if args.lookahead == True:
      gen_opt = Lookahead(gen_opt, alpha=args.lookahead_alpha)
      disc_opt = Lookahead(disc_opt, alpha=args.lookahead_alpha)

    # Initialize deepspeed
    gen_engine, gen_opt, *_ = deepspeed.initialize(args=args, model=gen, optimizer=gen_opt)
    disc_engine, disc_opt, *_ = deepspeed.initialize(args=args, model=disc, optimizer=disc_opt)

    device = gen_engine.local_rank
    batch_size = gen_engine.train_micro_batch_size_per_gpu()
    # TODO[?]: use rank/worldsize/?? from engine

    # Setup dataset and dataloaders
    dataset = Dataset(data_directory, image_size)
    num_workers = NUM_CORES if not is_ddp else 0
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True) if is_ddp else None
    dataloader = data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, sampler=sampler, shuffle=not is_ddp, drop_last=True, pin_memory=True)
    loader = cycle(dataloader)

    run = TrainingRun(
      gen=gen_engine, 
      gen_opt=gen_opt, 
      disc=disc_engine, 
      disc_opt=disc_opt, 
      args=args, 
      loader=loader, 
      batch_size=batch_size, 
      device=device)
    
    run.train()