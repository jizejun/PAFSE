# PAFSE

**PAFSE: Period-Aware Adaptive Frequency Separation Embedding for Temporal Knowledge Graph Completion**

本仓库实现 PAFSE，用于时序知识图谱补全（Temporal Knowledge Graph Completion, TKGC）。


---

## 1. 环境

论文中说明：

- Framework: **PyTorch**
- Optimizer: **Adagrad**
- GPU: **NVIDIA GeForce RTX 4090**
- Evaluation: **Filtered setting**
- Metrics: **MRR, Hits@1, Hits@3, Hits@10**

建议创建 Python 3.8+ 环境：

```bash
conda create -n pafse_env python=3.8 -y
conda activate pafse_env
conda install --file requirements.txt -c pytorch
```

`requirements.txt` 包含：

```text
tqdm
pytorch
numpy
scikit-learn
scipy
```

> 当前代码在训练阶段直接调用 `.cuda()`，因此按原代码运行需要 CUDA GPU。

---

## 2. 数据集

论文在三个标准 TKGC 数据集上进行实验：

| Dataset | Entities | Relations | Timestamps | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| GDELT | 500 | 20 | 366 | 2,735,685 | 31,961 | 31,961 |
| ICEWS14 | 7,128 | 230 | 365 | 72,826 | 8,963 | 8,941 |
| ICEWS05-15 | 10,488 | 251 | 4,017 | 386,962 | 46,092 | 46,275 |

原始数据目录应类似：

```text
src_data/
├── ICEWS14/
│   ├── train
│   ├── valid
│   └── test
├── ICEWS05-15/
│   ├── train
│   ├── valid
│   └── test
└── GDELT/
    ├── train
    ├── valid
    └── test
```

每行数据为：

```text
head_entity    relation    tail_entity    timestamp
```

即四列、使用 `\t` 分隔。

---

## 3. 数据预处理

ICEWS14 和 ICEWS05-15：

```bash
python process_icews.py
```

GDELT：

```bash
python process_gdelt.py
```

预处理后生成：

```text
data/
├── ICEWS14/
├── ICEWS05-15/
└── GDELT/
```

其中包含训练、验证、测试 pickle 文件以及 filtered ranking 使用的 `to_skip.pickle`。

---

## 4. 论文实验设置

论文通过 grid search 为不同数据集选择最终参数。

### 最终超参数

| Dataset | Paper subspace dimension \(d\) | `--rank` | Batch size | Max epochs | Learning rate | Optimizer | Frequency-loss weight \(\beta\) |
|---|---:|---:|---:|---:|---:|---|---:|
| ICEWS14 | 6000 | 6000 | 4000 | 5000 | 0.03 | Adagrad | 1e-5 |
| ICEWS05-15 | 8000 | 8000 | 6000 | 3000 | 0.03 | Adagrad | 1e-5 |
| GDELT | 8000 | 8000 | 2000 | 400 | 0.20 | Adagrad | 1e-5 |



## 5. 按论文参数训练

下面的命令匹配论文 **Experimental setup** 中明确报告的训练超参数。

### ICEWS14

论文设置：

```text
d = 6000
batch_size = 4000
max_epochs = 5000
learning_rate = 0.03
β = 1e-5
optimizer = Adagrad
```

运行：

```bash
python learner.py \
  --dataset ICEWS14 \
  --model PAFSE \
  --rank 6000 \
  --batch_size 4000 \
  --max_epochs 5000 \
  --learning_rate 0.03 \
  --use_freq_loss \
  --freq_beta 1e-5
```

---

### ICEWS05-15

论文设置：

```text
d = 8000
batch_size = 6000
max_epochs = 3000
learning_rate = 0.03
β = 1e-5
optimizer = Adagrad
```

运行：

```bash
python learner.py \
  --dataset ICEWS05-15 \
  --model PAFSE \
  --rank 8000 \
  --batch_size 6000 \
  --max_epochs 3000 \
  --learning_rate 0.03 \
  --use_freq_loss \
  --freq_beta 1e-5
```

---

### GDELT

论文设置：

```text
d = 8000
batch_size = 2000
max_epochs = 400
learning_rate = 0.20
β = 1e-5
optimizer = Adagrad
```

运行：

```bash
python learner.py \
  --dataset GDELT \
  --model PAFSE \
  --rank 8000 \
  --batch_size 2000 \
  --max_epochs 400 \
  --learning_rate 0.20 \
  --use_freq_loss \
  --freq_beta 1e-5
```

---



## 6. 结果保存

当前代码将结果保存到：

```text
results/<dataset>/<model>/
└── rank<rank>/
    └── lr<learning_rate>/
        └── batch<batch_size>/
            └── emb_reg<emb_reg>/
                └── time_reg<time_reg>/
                    └── freq_beta<freq_beta>/
```

例如按论文报告参数运行 ICEWS14，并使用当前代码默认 `emb_reg=0`、`time_reg=0`：

```text
results/ICEWS14/PAFSE/
└── rank6000/
    └── lr0.0300/
        └── batch4000/
            └── emb_reg0.00000/
                └── time_reg0.00000/
                    └── freq_beta1.0e-05/
```

主要输出文件：

```text
PAFSE.pkl
result+wuzaoting.txt
```

---

## 7. 一键复制：论文报告参数

### ICEWS14

```bash
python learner.py --dataset ICEWS14 --model PAFSE --rank 6000 --batch_size 4000 --max_epochs 5000 --learning_rate 0.03 --use_freq_loss --freq_beta 1e-5
```

### ICEWS05-15

```bash
python learner.py --dataset ICEWS05-15 --model PAFSE --rank 8000 --batch_size 6000 --max_epochs 3000 --learning_rate 0.03 --use_freq_loss --freq_beta 1e-5
```

### GDELT

```bash
python learner.py --dataset GDELT --model PAFSE --rank 8000 --batch_size 2000 --max_epochs 400 --learning_rate 0.20 --use_freq_loss --freq_beta 1e-5
```

---


