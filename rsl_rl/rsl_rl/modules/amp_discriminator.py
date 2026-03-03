import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd


class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim # amp_observation_dim * 2
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.LeakyReLU()) # amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device) # input and hidden layers 
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device) # output layer, last hidden to one

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        """Calculate the gradient penalty. Check gradient vanish or explosion."""
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True # auto calculate gradient on this tensor and stored in *.grad

        disc = self.amp_linear(self.trunk(expert_data)) # forward
        ones = torch.ones(disc.size(), device=disc.device)

        # obtain the gradient of discriminator
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        
        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()

        # printers
        # print('========================================')
        # print(disc.shape) # 8192,1
        # print(expert_state.shape) # 8192,67
        # print(expert_next_state.shape) # 8192,67
        # print(expert_data.shape) # 8192,134
        # print(grad.shape) # 8192,134
        # print(grad_pen) # single value

        return grad_pen
    

    def compute_logit_loss(self,
                         logit_reg_=0.01):
        """Calculate the logit_loss. """
        logit_weights = self.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        return logit_reg_ * disc_logit_loss
    
    def get_disc_logit_weights(self):
      return torch.flatten(self.amp_linear.weight)
    
    def compute_weight_pen(self, weight_decay_):

        weights = torch.cat(self.get_disc_weights(), dim=-1)
        weight_decay = torch.sum(torch.square(weights))
        return  weight_decay_ * weight_decay
    
    def get_disc_weights(self):
      weights = []
      for m in self.trunk.modules():
          if isinstance(m, nn.Linear):
              weights.append(torch.flatten(m.weight))

      weights.append(torch.flatten(self.amp_linear.weight))
      return weights
    

    # ===================================================================================
    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1))) # concatenate 2 states as discriminator input
            reward_d = torch.clamp(1-(1/4)*torch.square(d - 1), min=0)
            # eta = 0.35
            # reward = torch.tanh(eta * d) * 0.1
            # reward = torch.clamp(reward, min=-0.02) 
            # prob = 1 / (1 + torch.exp(-d)) 
            # reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            # reward = torch.exp(d)
            # reward *= self.amp_reward_coef
            reward_d *= self.amp_reward_coef
            x = torch.exp(-(1-reward_d)/0.25)
            if self.task_reward_lerp > 0: # if 0.0, only disc_reward
                reward = self._lerp_reward(x, task_reward.unsqueeze(-1))
                # reward = reward_d* task_reward.unsqueeze(-1)
                # reward = task_reward.unsqueeze(-1)*x
                # reward = (reward + 1) * (task_reward.unsqueeze(-1) + 1)
            self.train()
        return reward.squeeze(), reward_d
    
    # def predict_amp_reward(
    #         self, state, next_state, task_reward, normalizer=None):
    #     with torch.no_grad():
    #         self.eval()
    #         if normalizer is not None:
    #             state = normalizer.normalize_torch(state, self.device)
    #             next_state = normalizer.normalize_torch(next_state, self.device)

    #         d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1))) # concatenate 2 states as discriminator input
    #         # d = torch.clamp(d, max=1)
    #         # reward = torch.exp(d)

    #         reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0) # dist_reward, clamp to (0,1), better to be 1 and fast attenuate to 0

    #         if self.task_reward_lerp > 0: # if 0.0, only disc_reward
    #             reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
    #         self.train()
    #     return reward.squeeze(), d
    

    # def predict_amp_reward(
    #         self, state, next_state, task_reward, normalizer=None):
    #     with torch.no_grad():
    #         self.eval()
    #         if normalizer is not None:
    #             state = normalizer.normalize_torch(state, self.device)
    #             next_state = normalizer.normalize_torch(next_state, self.device)

    #         d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1))) # concatenate 2 states as discriminator input
    #         reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0) # dist_reward, clamp to (0,1), better to be 1 and fast attenuate to 0
            
    #         if self.task_reward_lerp > 0: # if 0.0, only disc_reward
    #             reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
    #         self.train()
    #     return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r