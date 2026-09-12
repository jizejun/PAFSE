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

在当前代码中，论文的子空间维度 \(d\) 对应：

```bash
--rank
```

模型内部使用 `2 * rank` 存储复值表示的实部与虚部，因此 README 中应传入论文给出的 \(d\)，而不是 `2d`。

论文明确指出，三个数据集的辅助频率分离损失权重均固定为：

```text
β = 1e-5
```

因此运行当前代码时必须显式加入：

```bash
--use_freq_loss --freq_beta 1e-5
```

因为 `learner.py` 中 `--freq_beta` 的代码默认值是 `1e-4`，与论文设置不同。

---

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

## 6. 论文中的训练目标

论文将 PAFSE 的主任务损失定义为：

```text
L_main = CE + λ R_N3
```

其中：

- `CE`：多分类 Cross Entropy；
- `R_N3`：N3 regularizer；
- `λ`：N3 正则系数。

辅助频率分离损失为：

```text
L_freq = -||X_L - X_H||_F + η ||X_H||_F
```

总损失为：

```text
L = L_main + β L_freq
```

论文实验设置明确给出：

```text
β = 1e-5
```

并使用：

```text
Adagrad
```

进行优化。

---

## 7. 关于 `--emb_reg` 和 `--time_reg`

当前代码还提供：

```bash
--emb_reg
--time_reg
```

其中：

- `--emb_reg` 控制代码中的 N3 regularizer；
- `--time_reg` 控制代码中的 `Lambda3` temporal regularizer。

需要注意：

1. 论文的训练目标明确包含 N3 正则项及其系数 \(\lambda\)，但当前论文的 **Experimental setup 并没有给出 \(\lambda\) 的具体数值**。
2. 论文给出的总目标中没有单独报告当前代码 `Lambda3` 对应的 `--time_reg` 数值。
3. 当前 `learner.py` 的代码默认值为：

```text
emb_reg = 0.0
time_reg = 0.0
```

因此，本 README 的论文复现命令没有把旧版 README 中的：

```text
ICEWS14:   emb_reg=0.01,  time_reg=0.01
ICEWS05-15: emb_reg=0.002, time_reg=0.1
GDELT:     emb_reg=0.001, time_reg=0.001
```

继续写成论文参数，因为这些数值**不属于当前 `PAFSE.tex` 的 Experimental setup 所报告的最终配置**。

如果需要严格复现论文表格中的最终数值，N3 正则系数 \(\lambda\) 仍需要由作者实验配置进一步确认。

---

## 8. 当前代码与论文设置的差异

为了避免把“代码行为”误写成“论文设置”，需要注意以下几点。

### 8.1 `freq_beta` 默认值不同

论文：

```text
β = 1e-5
```

当前代码默认：

```text
freq_beta = 1e-4
```

所以按论文训练时必须显式使用：

```bash
--freq_beta 1e-5
```

并同时开启：

```bash
--use_freq_loss
```

---

### 8.2 当前代码存在 50-epoch frequency-loss warm-up

`optimizers.py` 当前实现为：

```text
epoch < 50:
    L = L_fit + L_reg + L_time

epoch >= 50:
    L = L_fit + L_reg + L_time + β L_freq
```

也就是说，当前代码前 50 个 epoch 不把 frequency loss 加入总损失。

但是，论文的训练目标直接写为：

```text
L = L_main + β L_freq
```

论文 Experimental setup 中**没有说明前 50 个 epoch 关闭 frequency loss 的 warm-up 策略**。

因此：

- 本 README 的命令使用论文的 `β=1e-5`；
- 但直接运行当前代码时，仍会继承代码中的 50-epoch warm-up；
- 如果要求实现层面与论文公式完全一致，需要进一步统一这部分代码逻辑。

---

### 8.3 论文中的 \(\eta\)

论文频率损失写为：

```text
L_freq = -||X_L - X_H||_F + η ||X_H||_F
```

其中 \(\eta\) 是平衡高频激活的系数。

当前 `models.py` 中的 `loss_freq` 没有提供独立的命令行 `eta` 参数，对高频项采用固定实现。

因此 README 不把某个额外 `eta` 数值写成论文超参数，因为论文 Experimental setup 没有报告其具体取值。

---

## 9. 验证与早停

代码默认：

```text
valid_freq = 5
```

即每 5 个 epoch 进行一次验证。

可显式写为：

```bash
--valid_freq 5
```

当前早停逻辑的 patience 为 10 次验证。当 validation MRR 连续 10 次低于当前最佳值时提前终止。

> `valid_freq=5` 和 patience=10 是当前代码实现细节；论文 Experimental setup 没有将它们列为最终 grid-search 超参数。

---

## 10. 论文报告的 PAFSE 结果

论文在 standard filtered setting 下报告：

| Dataset | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|
| ICEWS14 | 0.647 | 0.565 | 0.695 | 0.800 |
| ICEWS05-15 | 0.698 | 0.618 | 0.748 | 0.844 |
| GDELT | 0.485 | 0.416 | 0.519 | 0.605 |

这些数值可作为复现实验的参考目标。

---

## 11. 论文中的超参数敏感性范围

论文还对学习率和 embedding dimension 进行了敏感性分析。

### Learning rate

ICEWS14 / ICEWS05-15：

```text
{0.01, 0.02, 0.03, 0.04, 0.05}
```

两者最优值均为：

```text
0.03
```

GDELT：

```text
{0.05, 0.10, 0.15, 0.20, 0.25}
```

最优值为：

```text
0.20
```

### Embedding / subspace dimension

论文测试：

```text
{2000, 4000, 6000, 8000}
```

最终配置：

```text
ICEWS14   -> 6000
ICEWS05-15 -> 8000
GDELT     -> 8000
```

---

## 12. 结果保存

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

## 13. 一键复制：论文报告参数

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

## 14. 复现前检查

```text
[ ] 使用 PyTorch + CUDA 环境
[ ] 已完成对应数据集预处理
[ ] ICEWS14 使用 rank=6000, batch=4000, epochs=5000, lr=0.03
[ ] ICEWS05-15 使用 rank=8000, batch=6000, epochs=3000, lr=0.03
[ ] GDELT 使用 rank=8000, batch=2000, epochs=400, lr=0.20
[ ] 三个数据集均启用 --use_freq_loss
[ ] 三个数据集均显式设置 --freq_beta 1e-5
[ ] 不把旧 README 的 emb_reg/time_reg 数值当成论文 Experimental setup 参数
[ ] 若要求“公式与实现完全一致”，进一步检查代码中的 50-epoch warm-up 和 η 实现
```
