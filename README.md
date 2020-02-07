## Pytorch Implementation of Unsupervised Data Augmentation for Consistency Training

```
python main.py \
    --dataset CIFAR10 \
    --num-labeled  1000 \
    --lr-warm-up  \
    --warm-up-steps 10000 \
    --num-steps 100000 \
    --batch-size-lab 64 \
    --batch-size-unlab 320 \
    --confidence-mask 0.6 \
    --softmax-temp 0.5 \
    --seed 3 \
    --rot \
    --verbose \
    --use-ema 
```
```
python main.py \
    --dataset ImageNet \
    --num-classes 1000 \
    --percent-labeled  10 \
    --lr-warm-up  \
    --warm-up-steps 10000 \
    --num-steps 200000 \
    --batch-size-lab 128 \
    --batch-size-unlab 128 \
    --lr 0.3 \
    --wdecay 1e-4 \
    --confidence-mask 0.5 \
    --softmax-temp 0.4 \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --seed 3 \
    --rot \
    --verbose \
    --use-ema 
```



Use **rot** flag for using rotation loss from S4L paper.
Run for 400k iterations(num-steps) for improved results.  

#### Results on CIFAR-10

| # Labels                   |  UDA Paper |  This UDA Repo  | with Rotation(rot) |
| -----------------          | ----  | ----- | ---- |
| 250                        | 8.76  | 11    |  8.2 |
| 500                        | 6.68  | 9.1   |  6.4 |
| 1000                       | 5.87  | 7.8   |  5.8 |
| 2000                       | 5.51  | 6.9   |  5.7 |
| 4000                       | 5.29  | 6.1   |  5.1 |

Model does not learn with higher learning rate lr=0.3
