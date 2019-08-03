import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_word_embeddings, num_char_embeddings,
                 kernels, num_input_channels, num_output_channels,
                 rnn_hidden_dim, num_layers, bidirectional,
                 word_padding_idx=0, char_padding_idx=0):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embeddings
        self.word_embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                            num_embeddings=num_word_embeddings,
                                            padding_idx=word_padding_idx)
        self.char_embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                            num_embeddings=num_char_embeddings,
                                            padding_idx=char_padding_idx)

        # Conv weights
        self.conv = nn.ModuleList([nn.Conv1d(num_input_channels,
                                             num_output_channels,
                                             kernel_size=f) for f in kernels])

        # GRU weights
        self.gru = nn.GRU(input_size=embedding_dim * (len(kernels) + 1),
                          hidden_size=rnn_hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)

    def initialize_hidden_state(self, batch_size, rnn_hidden_dim, device):
        """Modify this to condition the RNN."""
        num_directions = 1
        if self.bidirectional:
            num_directions = 2
        hidden_t = torch.zeros(self.num_layers * num_directions,
                               batch_size, rnn_hidden_dim).to(device)

    def get_char_level_embeddings(self, x):
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        batch_size, seq_len, word_len = input_shape
        x = x.view(-1, word_len)  # (N*seq_len, word_len)

        # Embedding
        x = self.char_embeddings(x)  # (N*seq_len, word_len, embedding_dim)

        # Rearrange input so num_input_channels is in dim 1 (N, embedding_dim, word_len)
        x = x.transpose(1, 2)

        # Convolution
        z = [F.relu(conv(x)) for conv in self.conv]

        # Pooling
        z = [F.max_pool1d(zz, zz.size(2)).squeeze(2) for zz in z]
        z = [zz.view(batch_size, seq_len, -1) for zz in z]  # (N, seq_len, embedding_dim)

        # Concat to get char-level embeddings
        z = torch.cat(z, 2)  # join conv outputs

        return z

    def forward(self, x_word, x_char, x_lengths, device):
        """
        x_word: word level representation (N, seq_size)
        x_char: char level representation (N, seq_size, word_len)
        """

        # Word level embeddings
        z_word = self.word_embeddings(x_word)

        # Char level embeddings
        z_char = self.get_char_level_embeddings(x=x_char)

        # Concatenate
        z = torch.cat([z_word, z_char], 2)

        # Feed into RNN
        initial_h = self.initialize_hidden_state(
            batch_size=z.size(0), rnn_hidden_dim=self.gru.hidden_size,
            device=device)
        out, h_n = self.gru(z, initial_h)

        return out