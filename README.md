## Accuracy
| Model             | Acc. 2-Step | Cosine Ann Acc. | Cos+ AutoAugment | Step + AA |
| ----------------- | ----------- | ---------       | ----- |  ---- |
| [ResNet18](https://arxiv.org/abs/1512.03385)      | 95.3  | 95.2  |  96.1  | 95.5  |
| WRN-28-2                                          | 94.7  | 94.7  |  95.9  | 95.0  | 
| WRN-28-10, 4k                                     |       |       |  83.6 (4k) | ---- | 

with Step Learning function | 0.05 | 0.01 | 0.002 |  95.7%
with Cosine Annealing Learning function | 0.04 | 96.0%

Ensemble of 8: 96.4 (lr period: 200 epochs)
Snapshot Ensemble of 8: 95.9 (lr period: 100 epochs)

