import deepspeed
from math import log2
import multiprocessing
import os
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
  def __init__(self, gen, gen_opt, disc, disc_opt, args, loader):
    self.gen = gen
    self.gen_opt = gen_opt
    self.disc = disc
    self.disc_opt = disc_opt
    self.loader = loader

    self.device = gen.local_rank
    self.batch_size = gen.train_micro_batch_size_per_gpu()

    self.image_size = args.image_size
    self.num_layers = int(log2(self.image_size) - 1)
    self.latent_dim = args.latent_dim
    self.lookahead = args.lookahead
    self.lookahead_k = args.lookahead_k
    self.checkpoint_every = args.checkpoint_every
    self.models_dir = args.models_dir
    self.results_dir = args.results_dir
    self.model_name = args.name
    self.gen_load_from = args.gen_load_from
    self.disc_load_from = args.disc_load_from

    self.rank = args.local_rank
    self.is_primary = self.rank == 0

    if self.is_primary:
      self.gen_ema = create_generator(args, self.device)
      self.gen_ema.requires_grad_(False)

      self.ema_k = args.ema_k
      self.ema = EMA(args.ema_beta)

    # TODO?: Stop keeping track of steps ourself, use the engine's.
    self.total_steps = 0
    self.mixed_prob = 0.9

    self.D_loss_fn = hinge_loss
    self.G_loss_fn = gen_hinge_loss
  
  def get_image_batch(self):
    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
    style = get_latents_fn(self.batch_size, self.num_layers, self.latent_dim, self.device)
    noise = image_noise(self.batch_size, self.image_size, self.device)

    return self.gen.forward(style, noise)
  
  def get_checkpoint_dirs(self):
      return (os.path.join(self.models_dir, self.model_name, 'gen'), 
        os.path.join(self.models_dir, self.model_name, 'disc'))
  
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

    # TODO: Probable bug with gradient_accumulation_steps and lookahead/ema updates.
    # Joint lookahead update
    if self.lookahead and (self.total_steps + 1) % self.lookahead_k == 0:
      self.gen_opt.lookahead_step()
      self.disc_opt.lookahead_step()

    # EMA update
    if self.is_primary and (self.total_steps + 1) % self.ema_k == 0:
      self.ema.update_ema(self.gen, self.gen_ema)

    self.total_steps = self.total_steps + 1
  
  def train(self):
    if self.is_primary:
      for _ in tqdm(forever()):
        # TODO: Check-ins (image dump & print to console)
        self.step()
        
        # Write checkpoint
        if self.total_steps % self.checkpoint_every == 0:
          gen_dir, disc_dir = self.get_checkpoint_dirs()

          if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
          if not os.path.exists(disc_dir):
            os.makedirs(disc_dir)
          
          self.gen.save_checkpoint(save_dir=gen_dir)
          self.disc.save_checkpoint(save_dir=disc_dir)
    else:
      while True:
        self.step()
  
  def load(self):
    # Load checkpoint, if it exists
    gen_dir, disc_dir = self.get_checkpoint_dirs()

    if (not os.path.exists(gen_dir)) or (not os.path.exists(disc_dir)):
      print("No model found to load. Starting fresh!")
      return
    
    print("Loading from checkpoint...")
    gen_tag = self.gen_load_from if len(self.gen_load_from) > 0 else None
    disc_tag = self.disc_load_from if len(self.disc_load_from) > 0 else None

    self.gen.load_checkpoint(load_dir=gen_dir, tag=gen_tag)
    self.disc.load_checkpoint(load_dir=disc_dir, tag=disc_tag)
    self.total_steps = self.gen.global_steps
    print(f"Resuming from step {self.total_steps}")

class Trainer():
  def __init__(self):
    pass
  
  def train(
    self,
    args,
  ):
    world_size = torch.distributed.get_world_size()
    is_ddp = world_size > 1
    rank = args.local_rank

    ttur_mult = 2.
    mixed_prob = 0.9

    gen = create_generator(args, rank)
    disc = stylegan2.AugmentedDiscriminator(args.image_size, network_capacity=args.network_capacity).cuda(rank)

    gen_opt = Adam(gen.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    disc_opt = Adam(disc.parameters(), lr=args.learning_rate * ttur_mult, betas=(0.5, 0.9))

    if args.lookahead == True:
      gen_opt = Lookahead(gen_opt, alpha=args.lookahead_alpha)
      disc_opt = Lookahead(disc_opt, alpha=args.lookahead_alpha)

    # Initialize deepspeed
    gen_engine, gen_opt, *_ = deepspeed.initialize(args=args, model=gen, optimizer=gen_opt)
    disc_engine, disc_opt, *_ = deepspeed.initialize(args=args, model=disc, optimizer=disc_opt)

    batch_size = gen_engine.train_micro_batch_size_per_gpu()
    num_workers = NUM_CORES if not is_ddp else 0

    # Setup dataset and dataloaders
    # TODO: Try passing dataset/loader into deepspeed initialize
    dataset = Dataset(args.data_dir, args.image_size)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True) if is_ddp else None
    dataloader = data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, sampler=sampler, shuffle=not is_ddp, drop_last=True, pin_memory=True)
    loader = cycle(dataloader)

    run = TrainingRun(
      gen=gen_engine, 
      gen_opt=gen_opt, 
      disc=disc_engine, 
      disc_opt=disc_opt, 
      args=args, 
      loader=loader)
    
    run.load()
    run.train()