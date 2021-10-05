
from collections import defaultdict
import torch

# lookahead
class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5):
        self.optimizer = optimizer
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)

    def lookahead_step(self):
        for group in self.param_groups:
            for fast in group["params"]:
                param_state = self.state[fast]
                if "slow_params" not in param_state:
                    param_state["slow_params"] = torch.zeros_like(fast.data)
                    param_state["slow_params"].copy_(fast.data)
                slow = param_state["slow_params"]
                # slow <- slow + alpha * (fast - slow)
                slow += (fast.data - slow) * self.alpha
                fast.data.copy_(slow)

    def step(self, closure = None):
        loss = self.optimizer.step(closure)
        return loss
