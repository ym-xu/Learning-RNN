import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder


class NewsModel(nn.Module):
    def __init__(self, embedding_dim, num_word_embeddings, num_char_embeddings,
                 kernels, num_input_channels, num_output_channels,
                 rnn_hidden_dim, hidden_dim, output_dim, num_layers,
                 bidirectional, dropout_p, word_padding_idx, char_padding_idx):
        super(NewsModel, self).__init__()
        self.encoder = Encoder(embedding_dim, num_word_embeddings,
                                   num_char_embeddings, kernels,
                                   num_input_channels, num_output_channels,
                                   rnn_hidden_dim, num_layers, bidirectional,
                                   word_padding_idx, char_padding_idx)
        self.decoder = Decoder(rnn_hidden_dim, hidden_dim, output_dim,
                                   dropout_p)

    def forward(self, x_word, x_char, x_lengths, device, apply_softmax=False):
        encoder_outputs = self.encoder(x_word, x_char, x_lengths, device)
        y_pred = self.decoder(encoder_outputs, apply_softmax)
        return y_pred