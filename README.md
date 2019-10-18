# Kitti- Road Segmentation
> Lane Segmentation using several architectures.

## I. Preparation:

- Install environment using Anaconda: `environment.yml`.

- Download dataset [here](), extract into `./data`.

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