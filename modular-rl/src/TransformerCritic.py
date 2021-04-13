from __future__ import print_function
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from TransformerActor import TransformerModel


class CriticTransformerPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super().__init__()
        self.num_limbs = 1
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.critic1 = TransformerModel(
            self.state_dim + action_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm,
        ).to(device)
        self.critic2 = TransformerModel(
            self.state_dim + action_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm,
        ).to(device)

    def forward(self, state, action):
        self.clear_buffer()

        assert (
            state.shape[1] == self.state_dim * self.num_limbs
        ), "state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}".format(
            self.state_dim * self.num_limbs,
            state.shape[1],
            self.num_limbs,
            self.state_dim,
        )

        self.input_state = state.reshape(self.batch_size, self.num_limbs, -1).permute(
            1, 0, 2
        )
        self.input_action = action.reshape(self.batch_size, self.num_limbs, -1).permute(
            1, 0, 2
        )

        inpt = torch.cat([self.input_state, self.input_action], dim=2)

        self.x1 = self.critic1(inpt)
        self.x2 = self.critic2(inpt)
        self.x1 = torch.squeeze(self.x1.permute(1, 0, 2))
        self.x2 = torch.squeeze(self.x2.permute(1, 0, 2))
        return self.x1, self.x2

    def Q1(self, state, action):
        self.clear_buffer()
        self.input_state = state.reshape(self.batch_size, self.num_limbs, -1).permute(
            1, 0, 2
        )
        self.input_action = action.reshape(self.batch_size, self.num_limbs, -1).permute(
            1, 0, 2
        )
        inpt = torch.cat([self.input_state, self.input_action], dim=2)
        self.x1 = torch.squeeze(self.critic1(inpt).permute(1, 0, 2))
        return self.x1

    def clear_buffer(self):
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        self.parents = parents
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
