from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    base_path = Path(__file__).parent.parent.absolute()
    seed: int = 3407 # seed suggested by arxiv.org/pdf/2109.08203.pdf
    gene_threshold: float = 0.5
    n_components: int = 50 # set it to 50 for larger dataset
    n_neighbors: int = 5 # k of k nearest neighbor alg.
    batch_size: int = 64
    n_epochs: int = 1001
    lr: float = 3e-3 # learning rate
    wd: float = 5e-4 # weight dacay
    ratio_train = 0.9 # ratio of training sample in data split
    ratio_val_to_test = 1.0 # validation to test ration in the remaining data
    masked_prob = 0.3 # ratio of masked genes to impute