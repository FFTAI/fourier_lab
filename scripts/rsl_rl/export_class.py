import copy
import os
import torch

class EncoderPolicyExpoter(torch.nn.Module):
    def __init__(self, policy):
        super(EncoderPolicyExpoter, self).__init__()
        if hasattr(policy, "fc_mu"):
            latent_dim = policy.fc_mu.out_features
        else:
            latent_dim = 0
        for k, l in policy.est_layers.items():
            latent_dim += l.out_features
        
        self.ob_dims = policy.actor[0].in_features - latent_dim
        self.encoder_dims = policy.encoder[0].in_features
        
        self.encoder = copy.deepcopy(policy.encoder)
        fc_mu = copy.deepcopy(policy.fc_mu)
        est_layer = copy.deepcopy(policy.est_layers)
        est_weights = [layer.weight.data for layer in est_layer.values()]
        est_bias = [layer.bias.data for layer in est_layer.values()]
        est_weights.append(fc_mu.weight.data)
        est_bias.append(fc_mu.bias.data)
        compos_est_layer = torch.nn.Linear(self.encoder[-2].out_features, latent_dim)
        compos_est_layer.weight.data = torch.cat(est_weights, dim=0)
        compos_est_layer.bias.data = torch.cat(est_bias, dim=0)
        enc_list = [e for e in self.encoder]
        enc_list.append(compos_est_layer)
        self.composite_encoder = torch.nn.Sequential(*enc_list)

        self.actor = copy.deepcopy(policy.actor)
    
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")

        policy_script = torch.jit.script(self)
        policy_script.save(os.path.join(path, f"policy.pt"))

        # actor_script = torch.jit.script(self.actor)
        # encoder_script = torch.jit.script(self.composite_encoder)
        # actor_script.save(os.path.join(path, f"backbone.pt"))
        # encoder_script.save(os.path.join(path, f"encoder.pt"))

    def forward(self, obs_h):
        z = self.composite_encoder(obs_h)
        obz = torch.cat([obs_h[:, -self.ob_dims:],z], dim=-1)
        action = self.actor(obz)
        return action