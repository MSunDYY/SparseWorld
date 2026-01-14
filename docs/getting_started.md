# Training and Testing Instructions
We recommend training with 4 or 8 GPUs, each with at least 24 GB of memory. Note that, for ease of reproducibility, we combine the pretraining and end-to-end fine-tuning into a single process, which slightly differs from the implementation details described in the paper.


**a. Train**
```
bash ./tools/dist_train.sh ./configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py 8
```

**b. Test**

Following the [PreWorld](https://github.com/getterupper/PreWorld) codebase, we evaluate occupancy forecasting and trajectory planning metrics separately. The occupancy prediction metrics are as follows:
```
python tools/test.py configs/sparseworld/nuscenes-temporal/sparse-occ-traj-finetune.py checkpoint.pth
```
This will output the mIoU and IoU scores, and also generate **output_data.pkl** in the **SparseWorld/admlp** directory, which stores the generated trajectories. 

Then, the planning scores are evaluated as follows:

```
cd AD-MLP/pytorch/admlp
python evaluate_for_mlp.py
```
**c. Visualization**

If you need occupancy visualization, please refer to [OPUS](https://github.com/jbwang1997/OPUS). It’s an excellent work that provides a clear and comprehensive visualization implementation, and much of our work is based on it.