import torch
from torch import nn
from torch_geometric.nn import GCNConv, Sequential

class Net(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                Sequential('x, edge_index', [
                    (GCNConv(in_ch, out_ch), 'x, edge_index -> x'),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )
        self.decoder = nn.ModuleList(
            [
                Sequential('x, edge_index', [
                    (GCNConv(in_ch, out_ch), 'x, edge_index -> x'),
                    nn.ReLU(),
                    nn.Dropout(),
                ][:None if out_ch!=channels[0] else -1])
                for in_ch, out_ch in zip(channels[-1::-1], channels[-2::-1])
            ]
        )
        # self.encoder = Sequential('x, edge_index', [
        #             (GCNConv(channels[0], channels[1]), 'x, edge_index -> x'),
        #             nn.ReLU(),
        #             nn.Dropout(),
        #         ])
        # self.decoder = Sequential('x, edge_index', [
        #             (GCNConv(channels[1], channels[0]), 'x, edge_index -> x'),
        #         ])
    def forward(self, x, edge_index):
        for enc in self.encoder:
            x = enc(x, edge_index)
        for dec in self.decoder:
            x = dec(x, edge_index)
        return x


if __name__ == "__main__":
    model = Net([128, 64, 32])
    print(model)