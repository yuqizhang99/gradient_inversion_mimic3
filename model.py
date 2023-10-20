import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, dim, dropout, num_classes=1,
                 depth=1, input_dim=76, **kwargs):

        super(LSTM, self).__init__()
        
        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.dropout = dropout
        self.depth = depth

        self.final_activation = nn.Sigmoid()

        self.masking = nn.Identity()  # Dummy layer, actual implementation of masking needed
        self.lstm_layers = nn.ModuleList()

        is_bidirectional = True
        
        for i in range(depth - 1):
            self.lstm_layers.append(nn.LSTM(input_dim if i == 0 else dim, 
                                            dim // 2 if is_bidirectional else dim, 
                                            batch_first=True, 
                                            bidirectional=is_bidirectional,
                                            dropout=dropout if i < depth - 2 else 0))  # Dropout is only applied between stacked layers

        self.output_lstm = nn.LSTM(dim if is_bidirectional else dim // 2, dim, batch_first=True, dropout=dropout)
        
        self.output_layer = nn.Sequential(
                nn.Linear(dim, num_classes),
                self.final_activation
            )

    def forward(self, x, m=None):
        x = self.masking(x)

        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        x, _ = self.output_lstm(x)
        
        x = x[:, -1, :]

        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)

        if m is not None:
            x = x * m.unsqueeze(-1).float()  # Applying the mask, if provided

        x = self.output_layer(x)

        return x