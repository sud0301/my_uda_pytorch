## Pytorch Implementation of Unsupervised Data Augmentation

```
python main.py \
    --lr-warm-up  \
    --num-labeled  1000 \
    --num-steps 100000 \
    --warm-up-steps 10000 \
    --batch-size-lab 64 \
    --batch-size-unlab 320 \
    --confidence-mask 0.6 \
    --softmax-temp 0.5 \
    --seed 3 \
    --rot 
```

#### Results on CIFAR-10

| # Labels                   |  UDA Paper |  UDA Repo  | UDA + Rotation |
| -----------------          | ----  | ----- | ---- |
| 250                        | 8.76  | 11    |  8.2 |
| 500                        | 6.68  | 9.1   |  7.7 |
| 1000                       | 5.87  | 7.8   |  5.8 |
| 2000                       | 5.51  | 6.9   |  6.1 |
| 4000                       | 5.29  | 6.1   |  5.1 |

