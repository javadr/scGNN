import numpy as np

from preprocessing import DataSanitizer, CellGraph
from config import CFG
from utils import Visualize, Runner
from pathlib import Path

from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from rich import print

# system inits
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)
np.random.seed(CFG.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if Path('processed.pt').exists():
    tload = torch.load('processed.pt')
    data, gd = tload['data'], tload['gd']
else:
    data = DataSanitizer('matrix.mtx', data_path='./data')
    gd = CellGraph(data.get(), device=device)[0]
    torch.save({'data': data, 'gd': gd}, 'processed.pt')

x, edge_index, n_features = gd.x, gd.edge_index, gd.num_features
print(gd)

data = torch.tensor(data.get(), dtype=torch.float).to(device)

for method in ('masked', 'full', 'zeros'):
    print(method.title().center(80, '='))

    n_features = gd.num_features
    model = Net([n_features, n_features//64]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)


    # base_loss = criterion(data[gd.train_mask], gd.x[gd.train_mask])
    runner = Runner(model, criterion, optimizer, x=x,
                    edge_index=edge_index, gd=gd, data=data)
    # CFG.n_epochs = 5001
    logs = runner.train_full(epochs=CFG.n_epochs, method=method)

    output, target, loss_test = runner.evaluate(method='zeros')
    print(f'Testing Loss: {loss_test:.4f} ')

    # output, target = run.predict()
    output, target = output.cpu().numpy().reshape(-1), target.cpu().numpy().reshape(-1)

    print(pearsonr(output, target))

    ext = f'(S{CFG.seed}E{CFG.n_epochs-1})'
    Visualize(logs).plot(
        title=f'Train Validation Loss #{method}', xlabel='#Epochs', ylabel='Loss Values', savefigname=f'train_val_{method}_{ext}.png')
    Visualize(output, target).regplot(
        title=f'neuron1K scGCN #{method}', xlabel='True', ylabel='Predicted', savefigname=f'neuron1k_correlation_{method}.png')
