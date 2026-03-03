# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.networks.memory import Memory
from rsl_rl.utils import split_and_pad_trajectories, my_split_and_pad_trajectories
import torch
import torch.nn as nn
from torch.distributions import Normal
from copy import deepcopy
from collections import OrderedDict

from rsl_rl.utils import resolve_nn_activation


class ActorCriticRecurrentVision(nn.Module):
    distribution: torch.distributions.Distribution
    is_recurrent = True

    def __init__(
        self,
        num_obs,
        num_obs_h,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_encoder_hidden_dims,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
        depth_obs_channels=1,
        vision_encoder_hidden_dims={
            'out_channels': [32, 64, 0],
            'kernels': [8, 4, 3],
            'strides': [4, 2, 1]
        },
        base_encoder_hidden_dims=[1024, 512, 256],
        transformer_num_layers=2,
        transformer_token_dim=128,
        transformer_num_heads=2,
        transformer_mlp_dim=512,
        estimation_dims={
            "velocity": [3],
            "height_map": [32, 128, 256, 99],
        },
        latent_dims=35,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.estimation_dims = estimation_dims
        self.num_actions = num_actions
        self.num_obs_h = num_obs_h
        self.latent_dims = latent_dims
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_encoder_layers = []
        actor_encoder_layers.append(nn.Linear(num_obs_h, actor_encoder_hidden_dims[0]))
        actor_encoder_layers.append(activation)
        for layer_index in range(len(actor_encoder_hidden_dims) - 1):
            if layer_index == len(actor_encoder_hidden_dims) - 2:
                actor_encoder_layers.append(nn.Linear(actor_encoder_hidden_dims[layer_index], actor_encoder_hidden_dims[layer_index + 1]))
            else:
                actor_encoder_layers.append(nn.Linear(actor_encoder_hidden_dims[layer_index], actor_encoder_hidden_dims[layer_index + 1]))
                actor_encoder_layers.append(activation)
        self.actor_encoder = nn.Sequential(*actor_encoder_layers)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # Base encoder
        modules = []
        _base_encoder_hidden_dims = base_encoder_hidden_dims + [transformer_token_dim] 
        modules.append(nn.Linear(num_obs_h, _base_encoder_hidden_dims[0]))
        modules.append(activation)
        for l in range(len(_base_encoder_hidden_dims) - 1):
            if l == len(_base_encoder_hidden_dims) - 2:
                modules.append(nn.Linear(_base_encoder_hidden_dims[l], _base_encoder_hidden_dims[l + 1]))
            else:
                modules.append(nn.Linear(_base_encoder_hidden_dims[l], _base_encoder_hidden_dims[l + 1]))
                modules.append(activation)
        self.base_encoder = nn.Sequential(*modules)

        #  Vision encoder
        out_channels = list(vision_encoder_hidden_dims['out_channels'])  # copy
        out_channels[-1] = transformer_token_dim  # last layer should match transformer token dimension
        kernels = vision_encoder_hidden_dims['kernels']
        strides = vision_encoder_hidden_dims['strides']
        assert len(out_channels) == len(kernels) == len(strides), "Vision encoder parameters must have the same length"

        conv_front = []
        conv_back = []
        in_channels = depth_obs_channels
        for oc, kernel_size, stride in zip(out_channels, kernels, strides):
            conv_front.extend([nn.Conv2d(in_channels, oc, kernel_size=kernel_size, stride=stride), activation])
            conv_back.extend([nn.Conv2d(in_channels, oc, kernel_size=kernel_size, stride=stride), activation])
            in_channels = oc
        self.vision_encoder_front = nn.Sequential(*conv_front)
        self.vision_encoder_back = nn.Sequential(*conv_back)

        # Transformer encoder — use TransformerEncoder for clarity
        encoder_layer = nn.TransformerEncoderLayer(
            transformer_token_dim, transformer_num_heads, transformer_mlp_dim,
            dropout=0.0, norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # Memory
        self.afore_memory_projector = nn.Linear(transformer_token_dim * 3, rnn_hidden_dim)
        self.memory = Memory(rnn_hidden_dim, type='gru', num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        # Last encodings
        self.last_latent_encoder = nn.Sequential(*[nn.Linear(rnn_hidden_dim, transformer_token_dim), activation]) 
        for est_name, dims in self.estimation_dims.items():
            setattr(self, f"{est_name}_encoder", nn.Linear(transformer_token_dim, dims[0]))

        # Decoders
        for est_name, dims in self.estimation_dims.items():
            if len(dims) > 1:
                modules = []
                _dims = dims[1:]
                modules.append(nn.Linear(dims[0], _dims[0]))
                modules.append(activation)
                for l in range(len(_dims) - 1):
                    if l == len(_dims) - 2:
                        modules.append(nn.Linear(_dims[l], _dims[l+1]))
                    else:
                        modules.append(nn.Linear(_dims[l], _dims[l + 1]))
                        modules.append(activation)
                setattr(self, f"{est_name}_decoder", nn.Sequential(*modules))

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of learnable parameters: {total_params / 1e6:.2f}M")

    def encode(self, obs_history, depth_obs, masks=None, hidden_states=None):
        """
        Encodes the input by passing through the encoder network
        and returns the latent encodings.
        """
        ### Computer the tokens seperately
        if masks is None:       # Inference mode
            s_token = self.base_encoder(obs_history)  # [batch_size, token_dim]
            front_vision_tokens = self.vision_encoder_front(depth_obs[..., 0].unsqueeze(1)) # [batch_size, token_dim, 4, 7]
            back_vision_tokens = self.vision_encoder_back(depth_obs[..., 1].unsqueeze(1))
            # print(f"depth_obs: {self.format_tensors(depth_obs[1, :10, 40, 0])}")
            # print(f"s_token: {self.format_tensors(s_token[1, :10])}")
            # print(f"front_vision_tokens: {self.format_tensors(front_vision_tokens[1, :, 0, 0])}")
            ### Fuze the tokens
            s_token = s_token.unsqueeze(0)  # [num_tokens, batch_size, token_dim]
            token_list = [s_token]

            vision_tokens = torch.cat(
                [front_vision_tokens.flatten(2).permute(2, 0, 1), 
                    back_vision_tokens.flatten(2).permute(2, 0, 1)],
                dim=0)

            token_list.append(vision_tokens)
            transformer_input = torch.cat(token_list, dim=0)

            ### Transformer encode
            transformer_out = self.transformer_encoder(transformer_input)

            transformer_out_state = transformer_out[0, ...]
            transformer_out_front_vision = transformer_out[1:29, ...].mean(dim=0)  #28=per_DepthView_TokenNums
            transformer_out_back_vision = transformer_out[29:, ...].mean(dim=0)  #28=per_DepthView_TokenNums
            
            rnn_input = self.afore_memory_projector(torch.cat([transformer_out_state, transformer_out_front_vision, transformer_out_back_vision], dim=1))
            # print(f"rnn_input: {self.format_tensors(rnn_input[1, :10])}")
        else:       # Batch mode
            pass

        rnn_out = self.memory(rnn_input, masks, hidden_states).squeeze(0)
        # print(f"rnn_out: {self.format_tensors(rnn_out[1, :20])}")

        ### Last encoder
        encoder_out = self.last_latent_encoder(rnn_out)
        # print(f"encoder_out: {self.format_tensors(encoder_out[1, :20])}")
        encodings = OrderedDict()
        for est_name, dims in self.estimation_dims.items():
            est_encoder = getattr(self, f"{est_name}_encoder")
            encodings[est_name] = est_encoder(encoder_out)
            
        return encodings

    def pre_encode(self, obs_history, depth_obs):
        """
        Pre-encodes the input by passing through the encoder network
        and returns the latent encodings for RNN processing.
        To avoid memory usage issues, this method does not need 'split and pad' input data.
        Shape of obs_history: [sequence_len, num_mini_batches, num_obs_h]
        """
        s_token = self.base_encoder(obs_history.flatten(0, 1))  # [batch_size * sequence_len, token_dim]
        front_vision_tokens = self.vision_encoder_front(depth_obs[..., 0].flatten(0, 1).unsqueeze(1))   # [batch_size * sequence_len, token_dim, 4, 7]
        back_vision_tokens = self.vision_encoder_back(depth_obs[..., 1].flatten(0, 1).unsqueeze(1))

        ### Fuze the tokens
        s_token = s_token.unsqueeze(0)  # [num_tokens, batch_size * sequence_len, token_dim]
        token_list = [s_token]

        vision_tokens = torch.cat(
        [front_vision_tokens.flatten(2).permute(2, 0, 1), 
            back_vision_tokens.flatten(2).permute(2, 0, 1)],
        dim=0)

        token_list.append(vision_tokens)
        transformer_input = torch.cat(token_list, dim=0)

        ### Transformer encode
        transformer_out = self.transformer_encoder(transformer_input)

        transformer_out_state = transformer_out[0, ...]
        transformer_out_front_vision = transformer_out[1:29, ...].mean(dim=0)  #28=per_DepthView_TokenNums
        transformer_out_back_vision = transformer_out[29:, ...].mean(dim=0)  #28=per_DepthView_TokenNums

        rnn_input = self.afore_memory_projector(torch.cat([transformer_out_state, transformer_out_front_vision, transformer_out_back_vision], dim=1))
        rnn_input = rnn_input.view(obs_history.shape[0], -1, rnn_input.shape[-1])

        return rnn_input  # [sequence_len, num_mini_batches, rnn_hidden_dim]   
    
    def post_encode(self, rnn_input, masks, hidden_states):
        """
        Post-encodes the RNN input and returns the latent encodings.
        This method is used after 'split and pad' the rnn_input.
        Shape of rnn_input: [sequence_len, num_mini_trajectories, rnn_hidden_dim]
        Shape of rnn_out: [sequence_len, num_mini_batches, rnn_hidden_dim]
        """
        rnn_out = self.memory(rnn_input, masks, hidden_states)

        ### Last encoder
        encoder_out = self.last_latent_encoder(rnn_out)
        encodings = OrderedDict()
        for est_name, dims in self.estimation_dims.items():
            est_encoder = getattr(self, f"{est_name}_encoder")
            encodings[est_name] = est_encoder(encoder_out)
            
        return encodings
    
    
    def decode(self, encodings):
        """
        Decodes the encodings using the decoders defined in the initialization.

        Args:
            encodings (dict): A dictionary where keys are estimation names and values are encoded tensors.

        Returns:
            dict: A dictionary where keys are estimation names and values are decoded tensors.
        """
        decodings = OrderedDict()
        for est_name, dims in self.estimation_dims.items():
            if len(dims) > 1:  # Only decode if the dimension length is greater than 1
                decoder = getattr(self, f"{est_name}_decoder", None)
                if decoder is not None:
                    decodings[est_name] = decoder(encodings[est_name])
                else:
                    raise AttributeError(f"Decoder for {est_name} not found. Ensure it is initialized in __init__.")
            else:   # If the dimension length is 1, directly assign the encoding
                decodings[est_name] = encodings[est_name]
        return decodings

    def ce_net(self, obs_history, depth_obs, dones=None, hidden_states=None):
        rnn_input = self.pre_encode(obs_history, depth_obs)
        masks, padded_trajectories = my_split_and_pad_trajectories(dones=dones, tensors=[rnn_input])
        sp_rnn_input = padded_trajectories[0]  # [sequence_len, num_mini_trajectories, rnn_hidden_dim]
        # print(f"sp_rnn_input.shape: {sp_rnn_input.shape}, masks.shape: {masks.shape}, hidden_states.shape: {hidden_states.shape}")
        encodings = self.post_encode(sp_rnn_input, masks, hidden_states)
        decodings = self.decode(encodings)
        return decodings
    
    def get_hidden_states(self):
        hidden_states = self.memory.hidden_states       # Gru's is not a tuple
        return hidden_states

    @property
    def has_implicit(self):
        """
        Check if the model has implicit parameters.

        Returns:
            bool: True if "obs_future" or other implicit estimation keys are present in `self.estimation_dims`, 
                  otherwise False.
        """
        # Check if "obs_future" or any other implicit-related key exists in estimation_dims
        return "implicit" in self.estimation_dims.keys()

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        self.memory.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        obs_h_encodings = self.actor_encoder(observations[..., :self.num_obs_h])
        mean = self.actor(torch.cat([obs_h_encodings, observations[..., self.num_obs_h:]], dim=-1))
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history, depth_obs, extras=None):
        encodings = self.encode(obs_history, depth_obs)
        hidden_states = self.get_hidden_states()
        # print(f"encodings:")
        # for name in encodings.keys():
        #     print(f"{name}:\n{self.format_tensors(encodings[name][1, :10])}")
        decodings = self.decode(encodings)
        # Calculate estimation loss if extras are provided
        if extras is not None:
            est_loss = {}
            for key in self.estimation_dims.keys(): 
                if key in decodings and key in extras:
                    est_err = nn.functional.mse_loss(decodings[key], extras[key])
                    est_loss[f"{key}_est_mse"] = est_err.item()
            # print(f"Estimation loss: {est_loss}")
        obs_h_encodings = self.actor_encoder(observations[..., :self.num_obs_h])
        decoding_values = [decodings[name] for name in self.estimation_dims.keys() if name in encodings]
        # decoding_values = [extras[name] for name in self.estimation_dims.keys() if name in encodings]
        self.log_inference(locals())

        actions_mean = self.actor(torch.cat([obs_h_encodings, *decoding_values], dim=-1))
        return actions_mean, decodings

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

    def format_tensors(self, tensors):
        tensors = tensors.tolist()
        formatted_states = [f"{v:.3f}" for v in tensors]
        columns_per_row = 10
        return "\n".join(
            [" ".join(formatted_states[i:i + columns_per_row]) for i in range(0, len(formatted_states), columns_per_row)]
        )

    def log_inference(self, local_vars):
        pass
        # print(f"obs_h_encodings:\n{self.format_tensors(local_vars['obs_h_encodings'][1, :20])}")
        # print(f"Hidden states:\n{self.format_tensors(local_vars['hidden_states'][0, 1, :10])}")
        # print(f"encodings:")
        # for name in local_vars['encodings'].keys():
            # print(f"{name}:\n{self.format_tensors(local_vars['encodings'][name][1, :10])}")
            # print(f"{name}:\n{self.format_tensors(local_vars['decodings'][name][1, :10])}")
