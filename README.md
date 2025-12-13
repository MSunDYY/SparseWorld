<div align="center">
<h3> [AAAI 2026] SparseWorld: A Flexible, Adaptive, and Efficient 4D Occupancy 

World Model Powered by Sparse and Dynamic Queries</h3>

<a href="https://arxiv.org/abs/2510.17482"><img src='https://img.shields.io/badge/arXiv-Paper-red' alt='Paper PDF'></a>

<div align="left">

## Abstract

Semantic occupancy has emerged as a powerful representation in world models for its ability to capture rich spatial
semantics. However, most existing occupancy world models rely on static and fixed embeddings or grids, which inherently limit the flexibility of perception. Moreover, their
“in-place classification” over grids exhibits a potential misalignment with the dynamic and continuous nature of real
scenarios. In this paper, we propose SparseWorld, a novel
4D occupancy world model that is flexible, adaptive, and efficient, powered by sparse and dynamic queries. We propose
a Range-Adaptive Perception module, in which learnable
queries are modulated by the ego vehicle states and enriched
with temporal-spatial associations to enable extended-range
perception. To effectively capture the dynamics of the scene,
we design a State-Conditioned Forecasting module, which
replaces classification-based forecasting with regressionguided formulation, precisely aligning the dynamic queries
with the continuity of the 4D environment. In addition, We
specifically devise a Temporal-Aware Self-Scheduling training strategy to enable smooth and efficient training. Extensive
experiments demonstrate that SparseWorld achieves state-ofthe-art performance across perception, forecasting, and planning tasks. Comprehensive visualizations and ablation studies further validate the advantages of SparseWorld in terms of
flexibility, adaptability, and efficiency.

<div align="left">

## Overview


<img src="./pics/overview.png" width="1000">
</div>

<div align="left">

## News
- **`2025/11/8`**: SparseWorld is accepted by AAAI 2026 🎉🎉!
- **`2025/10.10`**: The paper is released on [arXiv](https://arxiv.org/abs/2510.17482). 

## To Do
- [√] Release Paper
- [√] Release Code

## Getting Started
**a. Create a virtual environment**
```
cd SparseWorld
python3.9 -m venv env
source env/bin/activate
```

**b. Download some core packages**
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -v -e .
pip install -r requirements.txt
```

**c. Test**
```
python tools/test.py configs/preworld/nuscenes-temporal/sparse-occ-traj-finetune.py work_dirs/sparseworld-7frame-finetune-epoch56-new/epoch_56.pth
```

**d. Train**
```
bash ./tools/dist_train.sh ./configs/preworld/nuscenes-temporal/sparse-occ-traj-finetune.py 8
```

## Citation
If this work is helpful for your research, please consider citing:

```
@article{dang2025sparseworld,
  title={SparseWorld: A Flexible, Adaptive, and Efficient 4D Occupancy World Model Powered by Sparse and Dynamic Queries},
  author={Dang, Chenxu and Liu, Haiyan and Bao, Guangjun and An, Pei and Tang, Xinyue and Ma, Jie and Sun, Bingchuan and Wang, Yan},
  journal={arXiv preprint arXiv:2510.17482},
  year={2025}
}
```



## Acknowledge

Our code is developed based of following open source codebase:
- [OPUS](https://github.com/jbwang1997/OPUS)
- [PreWorld](https://github.com/getterupper/PreWorld)

We sincerely appreciate their outstanding works.