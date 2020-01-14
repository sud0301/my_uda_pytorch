## Pytorch Implementation of Unsupervised Data Augmentation for Consistency Training

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

