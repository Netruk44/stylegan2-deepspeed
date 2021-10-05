import torch

class EMA():
  def __init__(self, beta):
    super().__init__()
    self.beta = beta
    
  def update_ema(self, gen, gen_ema):
    param = list(gen.parameters())
    ema_param = list(gen_ema.parameters())

    for i in range(param):
      with torch.no_grad():
        ema_param[i].data.copy_(ema_param[i].data.mul(self.beta)
                           .add(param[i].data.mul(1-self.beta)))