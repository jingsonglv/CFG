
# CFG
This repository provides evaluation codes of CFG on ogbl-citation2 dataset for OGB link property prediction task. The idea of CFG is described in the following article:
> [Circle Feature Graphormer: Can Circle Features Stimulate Graph Transformer?](https://arxiv.org/abs/2309.06574)

This implementation of CFG for [**Open Graph Benchmak**](https://arxiv.org/abs/2005.00687) datasets (ogbl-citation2) is based on [**OGB**](https://github.com/snap-stanford/ogb) and [**SIEG**](https://github.com/anonymous20221001/SIEG_OGB). Thanks for their contributions.



## Requirements
The code is implemented with PyTorch and PyTorch Geometric. 

Requirments:  
1. python=3.7.11
2. pytorch=1.10.0
3. ogb=1.3.6
4. torch-geometric=1.7.2
5. dgl=0.8.1

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Install [Networkx](https://networkx.org/documentation/stable/install.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

Install [DGL](https://www.dgl.ai/pages/start.html)

Other required python libraries include: numpy, scipy, tqdm etc.

## Train and Predict
### ogbl-citation2:  

    python3 train.py --ngnn_code --grpe_cross --device 0  --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 8 --val_percent 4 --test_percent 0.2 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_swing

or 

    sh train_citation2.sh

#### CFG v2  

    python3 train.py --ngnn_code --grpe_cross --device 0  --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 8 --val_percent 4 --test_percent 0.2 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_bridge



## Results
The performances of CFG together with some selected GNN-based methods on OGB-CITATION2 task are listed as below:


| Method    | Test MRR  | Validation MRR  |
| ----------  | :-----------:  | :-----------: |
|  PLNLP    |  0.8492 ± 0.0029   |  0.8490 ± 0.0031       | 
| AGDN w/GraphSAINT  |  0.8549 ± 0.0029  |  0.8556 ± 0.0033        | 
| SEAL       |  0.8767 ± 0.0032        | 0.8757 ± 0.0031        |
| S3GRL (PoS Plus)  | 0.8814 ± 0.0008 | 0.8809 ± 0.0074 | 
| SUREL |  0.8883 ± 0.0018         |  0.8891 ± 0.0021       | 
| NGNN + SEAL   |  0.8891 ± 0.0022        |  0.8879 ± 0.0022          | 
| SIEG  |  0.8987 ± 0.0018 | 0.8978 ± 0.0018 | 
| CFG1   |  **0.8997 ± 0.0015** |  **0.8987 ± 0.0011** | 
| CFG2  |  **0.9003 ± 0.0007** |  **0.8992 ± 0.0007** | 


CFG achieves **top-1** performance on ogbl-citation2 in current OGB Link Property Prediction Leader Board until [**Sep 14, 2023**](https://ogb.stanford.edu/docs/leader_linkprop/). 


## License


CFG is released under an MIT license. Find out more about it [here](LICENSE).
