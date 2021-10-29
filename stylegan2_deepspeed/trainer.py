import deepspeed
from math import log2
import multiprocessing
import numpy as np
import os
from random import random
from stylegan2_deepspeed.dataset import Dataset
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

def cycle(iterable):
  while True:
    for i in iterable:
      yield i

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

    self.losses = {}

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

  def track_loss(self, name, loss):
    self.losses[name] = loss
  
  def get_training_image_batch(self):
    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
    style = get_latents_fn(self.batch_size, self.num_layers, self.latent_dim, self.device)
    noise = image_noise(self.batch_size, self.image_size, self.device)

    return self.gen.forward(style, noise)
  
  def get_checkpoint_dirs(self, create_if_needed=False):
    gen_path = os.path.join(self.models_dir, self.model_name, 'gen')
    disc_path = os.path.join(self.models_dir, self.model_name, 'disc')

    if create_if_needed:
      os.makedirs(gen_path, exist_ok=True)
      os.makedirs(disc_path, exist_ok=True)

    return (gen_path, disc_path)
  
  def gradient_penalty(self, images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch.autograd(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
  
  def step(self):
    # TODO: There's other loss calculations that still need to be copied from the original code.
    # apply_path_penalty, apply_gradient_penalty, 
    apply_gradient_penalty = self.current_step() % 4 == 0

    # Train Discriminator
    generated_images = self.get_training_image_batch()
    fake_output_loss, fake_q_loss = self.disc.forward(generated_images)

    image_batch = next(self.loader).cuda(self.device)
    image_batch.requires_grad_()
    real_output_loss, real_q_loss = self.disc.forward(image_batch)

    disc_loss = self.D_loss_fn(real_output_loss, fake_output_loss)
    
    if apply_gradient_penalty:
      gp = self.gradient_penalty(image_batch, real_output_loss)
      gp_val = gp.item()
      self.track_loss('GP', gp_val)
      disc_loss += gp

    disc_loss_value = disc_loss.item()
    self.track_loss('D', disc_loss_value)
    self.disc.backward(disc_loss)
    self.disc.step()

    # Train Generator
    generated_images = self.get_training_image_batch()
    fake_output_loss, _ = self.disc.forward(generated_images)
    real_output_loss = None

    gen_loss = self.G_loss_fn(fake_output_loss, real_output_loss)
    gen_loss_value = gen_loss.item()
    self.track_loss('G', gen_loss_value)
    self.gen.backward(gen_loss)
    self.gen.step()

    # Only run some things once per global_step.
    # Right at the very start of the next step, before we do any work.
    if self.current_microstep() == 0:
      # Joint lookahead update
      if self.lookahead and (self.current_step() + 1) % self.lookahead_k == 0:
        self.gen_opt.lookahead_step()
        self.disc_opt.lookahead_step()

      # EMA update
      if self.is_primary and (self.current_step() + 1) % self.ema_k == 0:
        self.ema.update_ema(self.gen, self.gen_ema)

  def current_iteration(self):
    # Essentially gen.micro_steps.
    # However, gen.micro_steps is reset to 0 at the start of every new training session, it's not loaded from the checkpoint.
    # So we have to calculate the current iteration instead.
    return self.gen.global_steps * self.gen.gradient_accumulation_steps() + self.current_microstep()

  def current_step(self):
    return self.gen.global_steps
  
  def current_microstep(self):
    return self.gen.micro_steps % self.gen.gradient_accumulation_steps()
  
  def train(self):
    iter = forever()

    if self.is_primary:
      iter = tqdm(iter, desc='Training')
    
    for _ in iter:
      # [All] Checkpoint
      if self.current_step() % self.checkpoint_every == 0 and self.current_microstep() == 0:
        gen_dir, disc_dir = self.get_checkpoint_dirs(create_if_needed=True)
        
        # TODO: Save EMA as part of generator checkpoint
        self.gen.save_checkpoint(save_dir=gen_dir)
        self.disc.save_checkpoint(save_dir=disc_dir)
      
      # [Primary] Generate training results
      if self.is_primary:
        if self.current_step() % self.evaluate_every == 0 and self.current_microstep() == 0:
          eval_id = self.current_step() // self.evaluate_every
          self.generate(eval_id)

      # [All] Step
      self.step()

      # [Primary] Update progress bar
      if self.is_primary:
        postfix = {'microstep': self.current_microstep(), 'step': self.current_step(), **self.losses}
        iter.set_postfix(postfix)

  def evaluate_in_chunks(self, gen, all_style, all_noise):
    imgs = []
    total_images = all_style[0][0].size(0)
    
    for start in range(0, total_images, self.batch_size):
      end = min(start + self.batch_size, total_images)

      # print(f'Generating {start} - {end} / {total_images}')
      style = [(s[0][start:end], s[1]) for s in all_style]
      noise = all_noise[start:end]
      next_images = gen.forward(style, noise)

      imgs.append(next_images)

    return torch.cat(imgs, dim=0)

  @torch.no_grad()
  def generate(self, eval_id):
    # Colab GDrive bug:
    # If you torchvision.utils.save_image() to a file that already exists in GDrive, it won't delete the old file.
    # It'll create a separate file with the same name alongside the old file. Because apparently that's a desirable feature?
    # Workaround: Delete the old file first.
    def save_image_delete_before_overwrite(images, output_file, nrow):
      if os.path.exists(output_file):
        os.remove(output_file)
      
      torchvision.utils.save_image(images, output_file, nrow=nrow)
    
    dest_dir = os.path.join(self.results_dir, self.model_name)
    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)

    num_rows = 8

    # Re-use latents across generators.
    total_latents = num_rows ** 2
    style = noise_list(total_latents, self.num_layers, self.latent_dim, self.device)
    noise = image_noise(total_latents, self.image_size, self.device)

    save_image_delete_before_overwrite(self.evaluate_in_chunks(self.gen, style, noise), os.path.join(dest_dir, f'{eval_id}.png'), num_rows)
    save_image_delete_before_overwrite(self.evaluate_in_chunks(self.gen_ema, style, noise), os.path.join(dest_dir, f'{eval_id}_ema.png'), num_rows)

    # Mixed latents / mixing regularities
    def tile(a, dim, n_tile):
      init_dim = a.size(dim)
      repeat_idx = [1] * a.dim()
      repeat_idx[dim] = n_tile
      a = a.repeat(*(repeat_idx))
      order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.device)
      return torch.index_select(a, dim, order_index)

    # Mix the latents of the first num_rows elements.
    style = style[0][0][:num_rows]
    tmp1 = tile(style, 0, num_rows)
    tmp2 = style.repeat(num_rows, 1)
    mixed_layer_count = self.num_layers // 2
    mixed_latents = [(tmp1, mixed_layer_count), (tmp2, self.num_layers - mixed_layer_count)]
    
    save_image_delete_before_overwrite(self.evaluate_in_chunks(self.gen, mixed_latents, noise), os.path.join(dest_dir, f'{eval_id}_mr.png'), num_rows)
    save_image_delete_before_overwrite(self.evaluate_in_chunks(self.gen_ema, mixed_latents, noise), os.path.join(dest_dir, f'{eval_id}_mr_ema.png'), num_rows)
  
  def load(self):
    # Load checkpoint, if it exists
    gen_dir, disc_dir = self.get_checkpoint_dirs()

    if (not os.path.exists(gen_dir)) or (not os.path.exists(disc_dir)):
      print("No model found to load. Starting fresh!")
      return
    
    # TODO: Check/override settings from constructor
    if self.is_primary:
      print("Loading from checkpoint...")

    gen_tag = self.gen_load_from if len(self.gen_load_from) > 0 else None
    disc_tag = self.disc_load_from if len(self.disc_load_from) > 0 else None

    # TODO: Load EMA as part of generator checkpoint
    self.gen.load_checkpoint(load_dir=gen_dir, tag=gen_tag)
    self.disc.load_checkpoint(load_dir=disc_dir, tag=disc_tag)

    if self.is_primary:
      print(f"Resuming from step {self.current_step()}, iteration: {self.current_iteration()}")

class Trainer():
  def __init__(self):
    pass
  
  def train(
    self,
    args,
  ):
    rank = args.local_rank
    ttur_mult = 2.

    gen = create_generator(args, rank)
    disc = stylegan2.AugmentedDiscriminator(args.image_size, network_capacity=args.network_capacity).cuda(rank)

    gen_opt = Adam(gen.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    disc_opt = Adam(disc.parameters(), lr=args.learning_rate * ttur_mult, betas=(0.5, 0.9))

    if args.lookahead == True:
      gen_opt = Lookahead(gen_opt, alpha=args.lookahead_alpha)
      disc_opt = Lookahead(disc_opt, alpha=args.lookahead_alpha)
    
    # Setup dataset
    dataset = Dataset(args.data_dir, args.image_size)

    # Initialize deepspeed
    gen_engine, gen_opt, dataloader, *_ = deepspeed.initialize(args=args, model=gen, optimizer=gen_opt, training_data=dataset)
    disc_engine, disc_opt, *_ = deepspeed.initialize(args=args, model=disc, optimizer=disc_opt)

    # Setup dataloaders
    loader = cycle(dataloader)

    run = TrainingRun(
      gen=gen_engine, 
      gen_opt=gen_opt, 
      disc=disc_engine, 
      disc_opt=disc_opt, 
      args=args, 
      loader=loader)
    
    if not args.new:
      run.load()
      
    run.train()