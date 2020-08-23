# Road segmentation - Autonomous car - FPT Digital Race (Cuộc Đua Số) 2020

This repository contains source code for road segmentation using ENet and UNet.

- Repository for car system: <https://github.com/vietanhdev/autonomous-car-2020>.
- Repository for road segmentation: <https://github.com/vietanhdev/autonomous-car-2020-road-seg>.
- Repository for traffic sign detection: <https://github.com/vietanhdev/autonomous-car-2020-sign-detection>.

## I. Preparation:

- Install environment using Anaconda: `environment.yml`.

- Download dataset, extract into `./data`.

## II. Training:

- Train the first time with general dataset

```
python3 train.py  --conf=./config_UNet.json
```

- Train again with dataset from Cuoc Dua So

```
python3 train.py  --conf=./config_UNet_final.json
```

## III. Run:

- Use `run.py`.


## NOTE: Run without GPU:

```
CUDA_VISIBLE_DEVICES="" <command>
```