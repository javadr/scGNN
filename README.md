# scGNN
scGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph.

A clean implementation of https://github.com/kexinhuang12345/scGNN in [`PyTorch`](https://pytorch.org/).

## Dataset
* [1k Brain Cells from an E18 Mouse (v3 chemistry)](https://www.10xgenomics.com/resources/datasets/1-k-brain-cells-from-an-e-18-mouse-v-3-chemistry-3-standard-3-0-0)

    Cells from a combined cortex, hippocampus and sub ventricular zone of an E18 mouse

## Reference
* [SCGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph Kexin](http://arxiv.org/abs/2008.03322)

### Requirements
The `scGNN` relies on `torch`, `torch-geometric`, `torch-cluster`, `networkx`, `rich`, `pandas`, `pathlib`, `sklearn`, and `scipy`.