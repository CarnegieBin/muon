# Muon Optimizer

Muon（MomentUm Orthogonalized by Newton-schulz）是一种用于大语言模型训练的优化器，通过对动量矩阵进行正交化（近似极分解）来实现谱范数方向上的最速下降。

本仓库提供两个版本的实现：

| 文件 | 正交化方法 | 适用硬件 | 精度 |
|------|-----------|---------|------|
| `muon.py` | 标准 Newton-Schulz | 所有 CUDA GPU | bfloat16 |
| `muon_gram.py` | Gram Newton-Schulz | 所有 CUDA GPU（含 H100 注释版） | float16 |

## 两个版本的区别

### 标准 Newton-Schulz (`muon.py`)

原始实现，来自 [KellerJordan/Muon](https://github.com/KellerJordan/Muon)。每次迭代在 n×m 矩形矩阵空间操作：

```
X_{t+1} = a*X_t + (b*A + c*A^2) @ X_t,  其中 A = X_t @ X_t^T
```

每步 FLOP：`2T(2α+1)n³`（α = m/n 为宽高比，T=5 次迭代）

### Gram Newton-Schulz (`muon_gram.py`)

来自 [Dao AI Lab](https://dao-lab.ai/blog/2026/gram-newton-schulz/) 的优化版本。核心思路：在 n×n Gram 矩阵空间迭代，最后再乘回 X：

```
polar(X) = (XX^T)^{-1/2} @ X
```

关键改进：
- **FLOP 减少 42-58%**：矩形乘法大幅减少，宽高比越大收益越大
- **Polar Express 系数**：比单组系数收敛更精确
- **Restart 策略**：第 2 次迭代后重启，消除 float16 下的数值不稳定
- **float16 替代 bfloat16**：值域在 1 附近，float16 精度更高
- **torch.baddbmm 融合运算**：使用 `torch.baddbmm` 将 `beta*A + alpha*(B@C)` 融合为单次 kernel 调用，减少中间张量分配和显存带宽消耗，在大矩阵上额外提速 ~10-15%

## 性能对比

以 T=5 次迭代为例：

| 宽高比 α | 标准 NS (无对称 GEMM) | Gram NS (稳定化) | FLOP 节省 |
|----------|---------------------|-----------------|----------|
| α = 2 | 50n³ | 23n³ | 54% |
| α = 4 | 90n³ | 41n³ | 54% |
| α = 8 | 170n³ | 77n³ | 55% |

实际 wall-clock 加速（取决于 cuBLAS 效率）：
- **A100/A800 (Ampere)**：NS 步骤加速 ~1.3-1.5×
- **H100/H800 (Hopper) + Quack kernel**：NS 步骤加速 ~2×

## 使用方法

### 基本用法

```python
from muon_gram import Muon, get_optimizer

# 方式一：直接使用 get_optimizer
optimizer = get_optimizer("muon", model, lr=1e-3, wd=0.1)

# 方式二：手动指定参数分组
muon_params = [
    p for name, p in model.named_parameters()
    if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
]
adamw_params = [
    p for name, p in model.named_parameters()
    if not (p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name)
]
optimizer = Muon(lr=1e-3, wd=0.1, muon_params=muon_params, adamw_params=adamw_params)
```

### 单独使用正交化函数

```python
# Gram Newton-Schulz（推荐）
from muon_gram import gram_newton_schulz
orthogonalized = gram_newton_schulz(gradient_matrix, steps=5)

# 标准 Newton-Schulz
from muon import zeropower_via_newtonschulz5
orthogonalized = zeropower_via_newtonschulz5(gradient_matrix, steps=5)
```

## 硬件兼容性

| GPU 架构 | 标准 NS (`muon.py`) | Gram NS (`muon_gram.py`) | Gram NS + Quack kernel |
|---------|--------------------|--------------------------|-----------------------|
| Ampere (A100/A800) | ✅ | ✅ | ❌ |
| Hopper (H100/H800) | ✅ | ✅ | ✅ |
| Blackwell (B200) | ✅ | ✅ | ✅ |

H100 用户如需启用 Quack 对称 GEMM kernel，参见 `muon_gram.py` 中注释掉的 `gram_newton_schulz_hopper` 函数，需安装 [Quack](https://github.com/Dao-AILab/quack)。

## 推荐配置

- **学习率**：0.02（配合 Moonshot LR scaling: `0.2 * sqrt(max(fan_out, fan_in))`）
- **动量**：0.95
- **NS 迭代次数**：5
- **Nesterov 动量**：开启
- **非 2D 参数**（embedding, lm_head, bias, norm）：使用内置 AdamW

## 参考

- [Muon 原始实现](https://github.com/KellerJordan/Muon)
- [Gram Newton-Schulz (Dao AI Lab)](https://dao-lab.ai/blog/2026/gram-newton-schulz/)
- [Polar Express 系数论文](https://arxiv.org/pdf/2505.16932)
- [Quack 对称 GEMM kernel](https://github.com/Dao-AILab/quack)
