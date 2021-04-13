import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math
import torch
import torch.nn as nn
from ModularActor import ActorGraphPolicy
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        feature_size,
        output_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
        condition_decoder=False,
        transformer_norm=False,
    ):
        """This model is built upon https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp) if transformer_norm else None,
        )
        self.encoder = nn.Linear(feature_size, ninp)
        self.ninp = ninp
        self.condition_decoder = condition_decoder
        self.decoder = nn.Linear(
            ninp + feature_size if condition_decoder else ninp, output_size
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        encoded = self.encoder(src) * math.sqrt(self.ninp)
        output = self.transformer_encoder(encoded)
        if self.condition_decoder:
            output = torch.cat([output, src], axis=2)

        output = self.decoder(output)

        return output


class TransformerPolicy(ActorGraphPolicy):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_action,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super(ActorGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = TransformerModel(
            self.state_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm,
        ).to(device)

    def forward(self, state, mode="train"):
        self.clear_buffer()
        if mode == "inference":
            temp = self.batch_size
            self.batch_size = 1

        self.input_state = state.reshape(self.batch_size, self.num_limbs, -1).permute(
            1, 0, 2
        )
        self.action = self.actor(self.input_state)
        self.action = self.max_action * torch.tanh(self.action)

        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
        self.action = self.action.permute(1, 0, 2)

        if mode == "inference":
            self.batch_size = temp

        return torch.squeeze(self.action)

    def change_morphology(self, parents):
        self.parents = parents
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
