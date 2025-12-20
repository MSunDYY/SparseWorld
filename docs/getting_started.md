

**a. Test**
```
python tools/test.py configs/preworld/nuscenes-temporal/sparse-occ-traj-finetune.py checkpoint.pth
```

**b. Train**
```
bash ./tools/dist_train.sh ./configs/preworld/nuscenes-temporal/sparse-occ-traj-finetune.py 8
```