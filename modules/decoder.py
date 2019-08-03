import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim, hidden_dim, output_dim, dropout_p):
        super(Decoder, self).__init__()

        # 注意力全连接模型
        self.fc_attn = nn.Linear(rnn_hidden_dim, rnn_hidden_dim)
        self.v = nn.Parameter(torch.rand(rnn_hidden_dim))

        # 全连接权重
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs, apply_softmax=False):

        # Attention
        z = torch.tanh(self.fc_attn(encoder_outputs))
        z = z.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        z = torch.bmm(v, z).squeeze(1)  # [B*T]
        attn_scores = F.softmax(z, dim=1)
        context = torch.matmul(encoder_outputs.transpose(-2, -1),
                               attn_scores.unsqueeze(dim=2)).squeeze()
        if len(context.size()) == 1:
            context = context.unsqueeze(0)

        # 全连接层
        z = self.dropout(context)
        z = self.fc1(z)
        z = self.dropout(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return attn_scores, y_pred