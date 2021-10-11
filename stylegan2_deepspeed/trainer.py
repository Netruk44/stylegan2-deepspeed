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
import torchvision
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
    assert log2(args.image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
    
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
    self.evaluate_every = args.evaluate_every
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
    else:
      self.gen_ema = None
      self.ema = None

    self.mixed_prob = 0.9

    self.D_loss_fn = hinge_loss
    self.G_loss_fn = gen_hinge_loss
  
  def get_image_batch(self, ema = False):
    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
    style = get_latents_fn(self.batch_size, self.num_layers, self.latent_dim, self.device)
    noise = image_noise(self.batch_size, self.image_size, self.device)

    generator = self.gen

    if ema:
      generator = self.gen_ema

    return generator.forward(style, noise)
  
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

    # Only run some things once per global_step.
    # Right at the very start of the step, before we do any work.
    if self.current_microstep() == 0:
      # Joint lookahead update
      if self.lookahead and (self.current_step() + 1) % self.lookahead_k == 0:
        self.gen_opt.lookahead_step()
        self.disc_opt.lookahead_step()

      # EMA update
      if self.is_primary and (self.current_step() + 1) % self.ema_k == 0:
        self.ema.update_ema(self.gen, self.gen_ema)

  def current_iteration(self):
    return self.gen.global_steps * self.gen.gradient_accumulation_steps() + self.current_microstep()

  def current_step(self):
    return self.gen.global_steps
  
  def current_microstep(self):
    return self.gen.micro_steps % self.gen.gradient_accumulation_steps()
  
  def train(self):
    iter = forever()

    if self.is_primary:
      iter = tqdm(iter)
    
    for _ in iter:
      # Checkpoint (all machines need to checkpoint)
      if self.current_step() % self.checkpoint_every == 0 and self.current_microstep() == 0:
        gen_dir, disc_dir = self.get_checkpoint_dirs()

        if not os.path.exists(gen_dir):
          os.makedirs(gen_dir)
        if not os.path.exists(disc_dir):
          os.makedirs(disc_dir)
        
        self.gen.save_checkpoint(save_dir=gen_dir)
        self.disc.save_checkpoint(save_dir=disc_dir)
      
      # Primary machine only
      if self.is_primary:
        # Generate results
        if self.current_step() % self.evaluate_every == 0 and self.current_microstep() == 0:
          eval_id = self.current_step() // self.evaluate_every
          print(f'Generating {eval_id}')
          self.generate(eval_id)
        else:
          print(f'Not generating {self.current_step()} % {self.evaluate_every} and {self.current_microstep()} == 0')

      self.step()

  @torch.no_grad()
  def generate(self, eval_id):
    def save_image_without_overwrite(images, output_file, nrow):
      if os.path.exists(output_file):
          os.remove(output_file)
      
      torchvision.utils.save_image(images, output_file, nrow=nrow)
    
    num_rows = 8
    all_imgs = []

    # Generate until we have enough to fill the grid
    while len(all_imgs) < num_rows ** 2:
      all_imgs = all_imgs + list(self.get_image_batch())
    
    # Only keep enough to fill the grid
    all_imgs = all_imgs[:num_rows**2]
    save_image_without_overwrite(all_imgs, os.path.join(self.results_dir, self.model_name, f'{eval_id}.png'), num_rows)

    # Repeat for EMA
    all_imgs = []
    while len(all_imgs) < num_rows ** 2:
      all_imgs = all_imgs + list(self.get_image_batch(ema=True))
    all_imgs = all_imgs[:num_rows**2]
    save_image_without_overwrite(all_imgs, os.path.join(self.results_dir, self.model_name, f'{eval_id}_ema.png'), num_rows)

    # TODO: Mixed latents

  
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
    print(f"Resuming from step {self.current_step()}, iteration: {self.current_iteration()}")

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