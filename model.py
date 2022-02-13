import torch
import torch.nn.functional as F
from torch import nn

class SentimentAnalysis(nn.Module):
    def __init__(self, vocab, out_channels, n_blocks, hidden_dim, num_layers, dropout, bidirectional, linear_dim):
        super(SentimentAnalysis, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(vocab["vectors"])

        ft_blocks = [FeatureBlock(300, out_channels)]
        for i in range(n_blocks-1):
            ft_blocks.append(FeatureBlock(3*out_channels, out_channels))
        self.ft_blocks = nn.Sequential(*ft_blocks)
        self.maxpool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=3*out_channels, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_channels=out_channels
        self.linear = nn.Sequential(
            nn.Linear((self.bidirectional+1)*hidden_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_dim, 1),
            nn.Sigmoid()
        )

    def init_lstm_hidden(self, bs):
        D=self.bidirectional+1
        h0 = torch.ones(D*self.num_layers, bs, self.hidden_dim).to(next(self.parameters()).device)
        c0 = torch.ones(D*self.num_layers, bs, self.hidden_dim).to(next(self.parameters()).device)
        return h0, c0

    def forward(self, x):
        x = self.embeddings(x).transpose(2, 1)
        x = self.ft_blocks(x)
        x = self.maxpool(x).transpose(2, 1)
        h0, c0 = self.init_lstm_hidden(x.shape[0])
        x, (h, c) = self.lstm(x, (h0 ,c0))
        x = self.linear(h[(-self.bidirectional-1):, :, :].transpose(1, 0).reshape(x.shape[0], -1))
        return x


class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureBlock, self).__init__()
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3)
    def forward(self, x):
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        return torch.cat((x3, x5, x7), axis=1)



