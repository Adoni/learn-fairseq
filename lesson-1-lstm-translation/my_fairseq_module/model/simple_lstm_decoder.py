# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/02
@desc: 这只飞很懒
"""
import torch
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from torch import nn


class SimpleLSTMDecoder(FairseqIncrementalDecoder):

    def __init__(
            self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128,
            dropout=0.1,
    ):
        # This remains the same as before.
        super().__init__(dictionary)
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )
        self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    # We now take an additional kwarg (*incremental_state*) for caching the
    # previous hidden and cell states.
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            # If the *incremental_state* argument is not ``None`` then we are
            # in incremental inference mode. While *prev_output_tokens* will
            # still contain the entire decoded prefix, we will only use the
            # last step and assume that the rest of the state is cached.
            prev_output_tokens = prev_output_tokens[:, -1:]

        # This remains the same as before.
        bsz, tgt_len = prev_output_tokens.size()
        final_encoder_hidden = encoder_out['final_hidden']
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout(x)
        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
            dim=2,
        )

        # We will now check the cache and load the cached previous hidden and
        # cell states, if they exist, otherwise we will initialize them to
        # zeros (as before). We will use the ``utils.get_incremental_state()``
        # and ``utils.set_incremental_state()`` helpers.
        initial_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )
        if initial_state is None:
            # first time initialization, same as the original version
            initial_state = (
                final_encoder_hidden.unsqueeze(0),  # hidden
                torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
            )

        # Run one step of our LSTM.
        output, latest_state = self.lstm(x.transpose(0, 1), initial_state)

        # Update the cache with the latest hidden and cell states.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', latest_state,
        )

        # This remains the same as before
        x = output.transpose(0, 1)
        x = self.output_projection(x)
        return x, None

    # The ``FairseqIncrementalDecoder`` interface also requires implementing a
    # ``reorder_incremental_state()`` method, which is used during beam search
    # to select and reorder the incremental state.
    def reorder_incremental_state(self, incremental_state, new_order):
        # Load the cached state.
        prev_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )

        # Reorder batches according to *new_order*.
        reordered_state = (
            prev_state[0].index_select(1, new_order),  # hidden
            prev_state[1].index_select(1, new_order),  # cell
        )

        # Update the cached state.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', reordered_state,
        )
