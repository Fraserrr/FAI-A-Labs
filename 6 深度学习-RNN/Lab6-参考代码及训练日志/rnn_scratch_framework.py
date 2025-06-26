"""
RNN From‑Scratch Framework（选做）
==========================
A pure‑NumPy implementation skeleton for a character‑level recurrent neural
network.  No deep‑learning libraries are used – students must complete missing
parts (marked TODO) to build forward and backward passes, train the model, and
sample text.

Run with:
    python rnn_scratch_framework.py
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import logging  # 新增导入
import time  # 新增导入用于生成日志文件名

import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm, trange  # 新增导入 tqdm
import pickle # 用于保存 c2i 和 i2c


# -----------------------------------------------------------------------------
# 0. 日志配置
# -----------------------------------------------------------------------------
def setup_logging():
    log_dir = "logs"  # 日志文件夹
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 当前时间戳
    log_file = os.path.join(log_dir, f"rnn_scratch_{timestamp}.log")  # 日志文件名

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,  # 日志级别
        format='%(asctime)s | %(levelname)s | %(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()  # 初始化日志记录器


# -----------------------------------------------------------------------------
# 1. Command‑line arguments
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NumPy RNN Shakespeare")
    p.add_argument("--seq_len", type=int, default=40, help="sequence length")
    p.add_argument("--hidden", type=int, default=128, help="hidden size")
    p.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    p.add_argument("--epochs", type=int, default=25, help="training epochs")
    p.add_argument("--batch", type=int, default=64, help="mini‑batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="shakespeare.txt")
    p.add_argument("--temp", type=float, default=1.0, help="sampling temperature")  # 为采样函数增加温度参数
    p.add_argument("--sample_length", type=int, default=200, help="length of sampled text")
    p.add_argument("--sample_seed", type=str, default="ROMEO: ", help="seed string for sampling")
    p.add_argument("--log_grad_norms", default=True, help="Log gradient norms for one batch periodically")
    p.add_argument("--log_grad_norms_epoch_interval", type=int, default=1, help="Epoch interval for logging gradient norms")
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2. Utils
# -----------------------------------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seed set to %d", seed)


def download_corpus(path):
    if Path(path).exists():
        logger.info("Corpus '%s' already exists.", path)
        return
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    logger.info("[+] downloading corpus from %s ...", url)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()  # 如果请求失败则抛出HTTPError
        Path(path).write_text(r.text, encoding="utf-8")
        logger.info("[✓] saved %s", path)
    except requests.exceptions.RequestException as e:
        logger.error("Failed to download corpus: %s", e)
        sys.exit(1)  # 下载失败则退出


# -----------------------------------------------------------------------------
# 3. Data preparation
# -----------------------------------------------------------------------------

def build_dataset(text, seq_len):
    chars = sorted(set(text))  # 获取所有不重复的字符并排序
    c2i = {c: i for i, c in enumerate(chars)}  # 字符到索引的映射
    i2c = {i: c for c, i in c2i.items()}  # 索引到字符的映射
    vocab_size = len(chars)  # 词汇表大小

    # 将整个语料库编码为整数数组
    encoded = np.array([c2i[c] for c in text], dtype=np.int32)

    # 创建序列和目标
    X, y = [], []
    # 滑动窗口，从0到 len(encoded) - seq_len -1
    # 每个X是长度为seq_len的序列，y是该序列后的一个字符
    for i in range(0, len(encoded) - seq_len):
        X.append(encoded[i: i + seq_len])
        y.append(encoded[i + seq_len])  # 目标是序列的下一个字符
    X = np.stack(X)  # (N, seq_len), N是样本数量
    y = np.array(y, dtype=np.int32)  # (N,)
    logger.info("Dataset created with %d sequences, vocab size %d", X.shape[0], vocab_size)
    return X, y, vocab_size, c2i, i2c


# -----------------------------------------------------------------------------
# 4. Model helpers
# -----------------------------------------------------------------------------

def one_hot(indices, depth):
    """将输入的整数索引转换为one-hot编码的2D数组。"""
    # indices: (batch_size, ) 或 (batch_size, 1) 的整数索引数组
    # depth: one-hot向量的维度，即词汇表大小
    out = np.zeros((indices.size, depth), dtype=np.float32)
    # np.arange(indices.size) 生成 0 到 batch_size-1 的序列
    # indices.flatten() 将indices展平为1D数组
    # 这两行使得在out数组的每一行，对应索引位置的元素被设为1.0
    out[np.arange(indices.size), indices.flatten()] = 1.0
    # 恢复原始indices的形状（除了最后一个维度是depth）
    return out.reshape(*indices.shape, depth)


class RNNCell:
    """单个时间步的Vanilla RNN单元 (tanh激活)。"""

    def __init__(self, input_size, hidden_size):
        # 参数初始化：使用 Xavier/Glorot 初始化思想，有助于缓解梯度消失/爆炸
        scale = 1.0 / np.sqrt(hidden_size)  # 通常是 1/sqrt(fan_in) 或 1/sqrt(fan_out)
        self.Wxh = np.random.randn(input_size, hidden_size) * scale  # 输入到隐藏层的权重
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale  # 隐藏层到隐藏层的权重
        self.bh = np.zeros(hidden_size)  # 隐藏层的偏置

        # 梯度 (在BPTT过程中缓存)
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

    def forward(self, x_t, h_prev):
        """
        RNN单元的前向传播。
        x_t: 当前时间步的输入 (B, I)，B是批量大小，I是输入维度 (词汇表大小)
        h_prev: 上一个时间步的隐藏状态 (B, H)，H是隐藏层大小
        返回:
            h_t: 当前时间步的隐藏状态 (B, H)
            cache: 用于反向传播的缓存 (x_t, h_prev, h_t)
        """
        # RNN核心公式: h_t = tanh(x_t @ Wxh + h_prev @ Whh + bh)
        h_raw = x_t @ self.Wxh + h_prev @ self.Whh + self.bh
        h_t = np.tanh(h_raw)
        cache = (x_t, h_prev, h_t)  # 缓存中间变量用于反向传播
        return h_t, cache

    def backward(self, dh_t, cache):
        """
        RNN单元的反向传播，计算一个时间步的参数梯度。
        dh_t: 上游传来的关于h_t的梯度 (B, H)
        cache: 前向传播时缓存的 (x_t, h_prev, h_t)
        返回:
            dh_prev: 关于h_prev的梯度，用于在时间序列上传播 (B, H)
        """
        x_t, h_prev, h_t = cache  # 从缓存中取出变量

        # 计算tanh的导数: (1 - h_t^2)
        # dtanh 是 d(loss)/d(h_raw) = d(loss)/d(h_t) * d(h_t)/d(h_raw)
        dtanh = dh_t * (1.0 - h_t ** 2)  # (B, H)

        # 参数梯度在序列上累加
        # dWxh = x_t.T @ dtanh
        self.dWxh += x_t.T @ dtanh
        # dWhh = h_prev.T @ dtanh
        self.dWhh += h_prev.T @ dtanh
        # dbh = sum(dtanh, axis=0)
        self.dbh += dtanh.sum(axis=0)

        # 计算传递到上一个时间步的隐藏状态的梯度
        # dh_prev = dtanh @ Whh.T
        dh_prev = dtanh @ self.Whh.T
        return dh_prev

    def zero_grad(self):
        """将存储的梯度清零。"""
        self.dWxh.fill(0)
        self.dWhh.fill(0)
        self.dbh.fill(0)

    def step_grad(self, lr):
        """根据计算得到的梯度更新参数。"""
        for param, grad in (
                (self.Wxh, self.dWxh),
                (self.Whh, self.dWhh),
                (self.bh, self.dbh),
        ):
            param -= lr * grad  # 梯度下降


class RNN:
    """字符级RNN模型，带有softmax输出 (单隐藏层)。"""

    def __init__(self, vocab_size, hidden_size):
        self.cell = RNNCell(vocab_size, hidden_size)  # RNN单元
        # 输出层权重
        scale = 1.0 / np.sqrt(hidden_size)
        self.Why = np.random.randn(hidden_size, vocab_size) * scale  # 隐藏层到输出层的权重
        self.by = np.zeros(vocab_size)  # 输出层的偏置
        # 梯度
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def zero_grad(self):
        """将模型所有参数的梯度清零。"""
        self.cell.zero_grad()
        self.dWhy.fill(0)
        self.dby.fill(0)

    # ---------------------------------------------------------------------
    # TODO 1: 实现整个序列的前向传播。
    # 输入:
    #   X_batch: (B, T) 整数索引的批次数据, B是批量大小, T是序列长度
    #   h0:      (B, H) 初始隐藏状态, H是隐藏层大小
    # 应返回 (logits, h_T, caches)
    #   logits:  (B, V) 最后一个时间步的分数, V是词汇表大小
    #   h_T:     (B, H) 最后一个时间步的隐藏状态
    #   caches:  用于反向传播的每一步的缓存列表
    # ---------------------------------------------------------------------
    def forward(self, X_batch, h0):
        """
        RNN模型在整个序列上的前向传播。
        """
        B, T = X_batch.shape  # B: 批量大小, T: 序列长度
        V = self.by.size  # V: 词汇表大小
        # H = h0.shape[1] # H: 隐藏层大小 (注释掉，因为cell中已有hidden_size信息)

        h_t = h0  # 初始化隐藏状态
        caches = []  # 存储每个时间步的缓存

        # 遍历序列中的每个时间步
        for t in range(T):
            # X_batch[:, t] 取出当前时间步所有批次的字符索引 (B,)
            # one_hot将其转换为 (B, V) 的one-hot编码
            x_t_one_hot = one_hot(X_batch[:, t], V)
            # RNN单元的前向传播
            h_t, cache = self.cell.forward(x_t_one_hot, h_t)
            caches.append(cache)  # 保存当前时间步的缓存

        # 计算最后一个时间步的输出logits
        # 实验指导要求仅用 h_T, 即最后一个时间步的隐藏状态来预测
        # logits = h_t @ Why + by
        logits = h_t @ self.Why + self.by  # (B, V)
        return logits, h_t, caches  # h_t此时即为h_T

    # ---------------------------------------------------------------------
    # TODO 2: 实现BPTT (Backpropagation Through Time)。
    # 输入:
    #   dlogits: (B, V) 关于分数的梯度
    #   caches:  forward方法返回的缓存列表
    #   h_T:     最后一个时间步的隐藏状态 (在前向传播中已计算)
    # 返回 dh0 (关于初始隐藏状态的梯度)
    # ---------------------------------------------------------------------
    def backward(self, dlogits, caches, h_T, collect_grad_norms=False):
        """
        RNN模型的BPTT反向传播。
        collect_grad_norms: 是否收集并返回每个时间步的隐藏状态梯度范数
        """

        # 输出层梯度计算
        self.dWhy += h_T.T @ dlogits
        self.dby += dlogits.sum(axis=0)

        # 传递到最后一个隐藏层的梯度
        dh = dlogits @ self.Why.T  # (B, H) 这是dL/dh_T

        dh_norms_over_time = []  # 用于存储梯度范数

        if collect_grad_norms:
            # dh 此时是 dL/dh_T (最后一个时间步的隐藏层输出梯度)
            # 我们通常关心的是从后往前传播时，更早时间步的梯度如何变化
            # 所以我们从 caches 的最后一个开始记录，对应于序列的最后一个时间步
            # dh_norms_over_time.append(np.linalg.norm(dh, axis=1).mean()) # 记录dL/dh_T的范数（批次平均）
            pass  # 或者从第一个通过cell反向传播的梯度开始记录

        # 沿时间反向传播
        # caches的顺序是从 t=0 到 t=T-1
        # reversed(caches) 是从 t=T-1 到 t=0
        # enumerate(reversed(caches))会给出 (0, cache_T-1), (1, cache_T-2), ...
        # 我们想将梯度范数与原始时间步t（从0到T-1）对应起来

        # 记录 dh (即 dL/dh_t) 的范数
        # 这个 dh 是输入到 RNNCell.backward 的梯度
        # 对应于公式中的 delta_t (或者说与delta_t密切相关的 dL/dh_t)

        # 调整收集逻辑：我们在每次调用 cell.backward *之前* 记录 dh 的范数
        # dh_t 是指当前时间步 t 的隐藏状态梯度 dL/dh_t
        # BPTT 是从后往前，所以第一个遇到的 dh 是 dL/dh_T

        temp_dh_for_norm_tracking = dh.copy()  # 初始的 dh 是 dL/dh_T

        for t_rev, cache in enumerate(reversed(caches)):  # t_rev = 0 corresponds to last timestep T-1
            # cache is (x_t, h_prev_t, h_t)
            if collect_grad_norms:
                # temp_dh_for_norm_tracking 是 dL/dh_{T-1-t_rev}
                # 我们从后往前记录，所以第一个是 dL/dh_T，然后 dL/dh_{T-1} ... dL/dh_1
                # axis=1 对每个样本计算范数，然后取平均值
                dh_norm = np.linalg.norm(temp_dh_for_norm_tracking, axis=1).mean()
                dh_norms_over_time.append(dh_norm)

            temp_dh_for_norm_tracking = self.cell.backward(temp_dh_for_norm_tracking, cache)

        # 由于上面循环中 temp_dh_for_norm_tracking 在循环结束后是 dL/dh_0
        # 而我们记录的是每次进入 cell.backward 之前的 dh_t
        # 所以 dh_norms_over_time 的顺序是 [norm(dL/dh_T), norm(dL/dh_{T-1}), ..., norm(dL/dh_1)]
        # 如果需要按时间顺序 t=0...T-1，则需要反转它
        if collect_grad_norms:
            dh_norms_over_time.reverse()  # 现在是 [norm(dL/dh_1), ..., norm(dL/dh_T)]
            # 注意：严格的delta_t可能还包含 (1-h_t^2) 部分，
            # 但 dL/dh_t 的范数变化也能很好地反映梯度传播情况。
            # 我们的 dh 是 dL/dh_t，在cell.backward内部会乘以 (1-h_t^2)

        dh0 = temp_dh_for_norm_tracking  # 这是 dL/dh_0

        if collect_grad_norms:
            return dh0, dh_norms_over_time
        else:
            return dh0

    def step_grad(self, lr):
        """根据计算的梯度更新模型所有参数。"""
        self.cell.step_grad(lr)  # 更新RNN单元的参数
        # 更新输出层的参数
        for param, grad in ((self.Why, self.dWhy), (self.by, self.dby)):
            param -= lr * grad


# -----------------------------------------------------------------------------
# 5. Loss helpers
# -----------------------------------------------------------------------------

def softmax_cross_entropy(logits, targets):
    """
    计算softmax交叉熵损失和梯度 (向量化实现)。
    logits: 模型的原始输出分数 (B, V)
    targets: 真实的目标类别索引 (B,)
    返回:
        loss: 平均交叉熵损失 (标量)
        grad: 关于logits的梯度 (B, V)
    """
    # 数值稳定性技巧：logits减去其最大值，避免exp溢出
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (B, V) softmax概率

    N = targets.size  # 批量大小
    # 交叉熵损失: -log(p_target)
    # probs[np.arange(N), targets] 选取每个样本对应目标类别的概率
    loss = -np.log(probs[np.arange(N), targets]).mean()  # 计算平均损失

    # 梯度计算: dL/dlogits = probs - y_one_hot
    # y_one_hot 在目标类别位置为1，其余为0
    # grad 初始化为 probs
    grad = probs.copy()  # 必须用copy，否则会修改probs
    grad[np.arange(N), targets] -= 1.0  # 在目标类别位置减1
    grad /= N  # 平均梯度
    return loss, grad


# -----------------------------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------------------------

def train(model, X, y, epochs, lr, batch_size, hidden_size, args): # 添加args
    N = X.shape[0]
    history = []

    epoch_bar = trange(1, epochs + 1, desc="Training", unit="epoch")
    for ep in epoch_bar:
        idx = np.random.permutation(N)  # 每个epoch开始时打乱数据顺序
        X_shuffled, y_shuffled = X[idx], y[idx]
        ep_loss = 0.0

        # 使用tqdm创建batch的进度条
        num_batches = N // batch_size
        batch_bar = tqdm(range(num_batches), desc=f"Epoch {ep}/{epochs}", leave=False, unit="batch")

        for i in batch_bar:
            start = i * batch_size
            end = start + batch_size
            xb = X_shuffled[start:end]  # 当前批次的输入序列
            yb = y_shuffled[start:end]  # 当前批次的目标字符

            # 为简单起见，如果最后一个批次大小不足，则跳过 (原始代码逻辑保留)
            # 更好的做法是处理不完整的批次，但这会增加代码复杂性
            if xb.shape[0] != batch_size:  # 严格来说，这在上面num_batches处已经避免了
                continue

            # 初始化隐藏状态为零向量
            h0 = np.zeros((batch_size, hidden_size), dtype=np.float32)

            # 前向传播
            logits, h_T, caches = model.forward(xb, h0)
            # 计算损失和梯度
            loss, dlogits = softmax_cross_entropy(logits, yb)

            # 反向传播前梯度清零
            model.zero_grad()

            collect_norms_this_batch = False
            if args.log_grad_norms and ep % args.log_grad_norms_epoch_interval == 0 and i == 0:  # 例如，每隔N个epoch的第一个batch
                collect_norms_this_batch = True

            # 根据是否收集范数，调用不同版本的 backward
            if collect_norms_this_batch:
                _, grad_norms = model.backward(dlogits, caches, h_T, collect_grad_norms=True)
                logger.info(f"Epoch {ep}, Batch 0, Gradient norms dL/dh_t (t=1 to T, batch avg):")
                # 打印范数，可以每隔几个时间步打印一个，避免过多输出
                log_str = []
                for t_idx, norm_val in enumerate(grad_norms):  # grad_norms 现在是按时间顺序 t=1..T
                    if t_idx % (max(1, len(grad_norms) // 10)) == 0 or t_idx == len(grad_norms) - 1:  # 每隔10%或最后一个
                        log_str.append(f"t={t_idx + 1}: {norm_val:.4e}")  # t_idx从0开始，所以时间步是t_idx+1
                logger.info("    " + " | ".join(log_str))

            else:
                model.backward(dlogits, caches, h_T, collect_grad_norms=False)

            model.step_grad(lr)
            ep_loss += loss * xb.shape[0]
            batch_bar.set_postfix(loss=f"{loss:.4f}")

        ep_loss /= (num_batches * batch_size)
        history.append(ep_loss)
        epoch_bar.set_postfix(mean_loss=f"{ep_loss:.4f}")
        logger.info(f"Epoch {ep}/{epochs}  Mean Loss: {ep_loss:.4f}")
    return history


# -----------------------------------------------------------------------------
# 7. Sampling
# -----------------------------------------------------------------------------

def sample(model, seed_str, c2i, i2c, vocab_size, length, temp, hidden_size, seq_len_for_sampling):
    """
    从训练好的模型生成文本。
    model: 训练好的RNN模型。
    seed_str: 初始种子字符串。
    c2i, i2c: 字符与索引的映射。
    vocab_size: 词汇表大小。
    length: 要生成的文本长度。
    temp: 采样温度。
    hidden_size: 模型的隐藏层大小。
    seq_len_for_sampling: 采样时模型输入的序列长度 (应与训练时一致或兼容)。
    """
    # 将种子字符串转换为索引序列
    seq_indices = [c2i[c] for c in seed_str if c in c2i]
    if not seq_indices:  # 如果种子字符串中没有已知字符，则随机选择一个起始字符
        logger.warning("Seed string contains no known characters. Starting with a random char.")
        start_char_idx = np.random.randint(vocab_size)
        seq_indices = [start_char_idx]
        seed_str = i2c[start_char_idx]

    h = np.zeros((1, hidden_size), dtype=np.float32)  # 初始化隐藏状态 (批次大小为1)
    output_text = seed_str  # 输出文本以种子字符串开始

    # logger.info("Sampling with seed: '%s'", seed_str)

    # 逐字符生成
    for _ in range(length):
        # 准备当前输入序列 (取末尾seq_len_for_sampling个字符作为输入)
        # 如果当前seq_indices长度不足seq_len_for_sampling，则用0填充 (或选择其他填充策略)
        current_input_indices = np.array(seq_indices[-seq_len_for_sampling:], dtype=np.int32).reshape(1, -1)

        # 如果 current_input_indices 长度不足 seq_len_for_sampling，需要填充
        # 这种填充方式可能不是最优的，但能保证输入形状正确
        if current_input_indices.shape[1] < seq_len_for_sampling:
            padding = np.zeros((1, seq_len_for_sampling - current_input_indices.shape[1]), dtype=np.int32)
            current_input_indices = np.concatenate((padding, current_input_indices), axis=1)

        # 模型前向传播
        logits, h, _ = model.forward(current_input_indices, h)  # caches在此处不需要

        # 应用温度进行缩放
        logits /= temp
        # 计算概率分布 (softmax)
        # 减去最大值以保证数值稳定性
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()  # (1, V)

        # 从概率分布中随机选择下一个字符的索引
        next_char_idx = np.random.choice(vocab_size, p=probs.ravel())

        # 将生成的字符添加到输出文本和索引序列中
        output_text += i2c[next_char_idx]
        seq_indices.append(next_char_idx)

    return output_text


# -----------------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()  # 解析命令行参数
    set_seed(args.seed)  # 设置随机种子
    download_corpus(args.data)  # 下载语料库

    text = Path(args.data).read_text(encoding="utf-8")  # 读取文本数据
    # 构建数据集
    X, y, vocab_size, c2i, i2c = build_dataset(text, args.seq_len)

    # 初始化模型
    model = RNN(vocab_size, args.hidden)
    logger.info("Model initialized: RNN(vocab_size=%d, hidden_size=%d)", vocab_size, args.hidden)
    logger.info("Training parameters: epochs=%d, lr=%.4f, batch_size=%d, seq_len=%d",
                args.epochs, args.lr, args.batch, args.seq_len)

    # 训练模型
    losses = train(
        model, X, y, args.epochs, args.lr, args.batch, args.hidden, args
    )

    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss (NumPy RNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    fig_filename = f"loss_numpy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_filename)
    logger.info("[✓] Loss curve saved to %s", fig_filename)

    # --- 保存模型参数和元数据 ---
    model_save_path = f"numpy_rnn_model"
    weights_path = model_save_path + ".npz"
    metadata_path = model_save_path + "_meta.pkl"

    np.savez_compressed(weights_path,
                        Why=model.Why,
                        by=model.by,
                        Wxh=model.cell.Wxh,
                        Whh=model.cell.Whh,
                        bh=model.cell.bh)
    logger.info("[✓] Model weights saved to %s", weights_path)

    metadata = {
        'vocab_size': vocab_size,
        'hidden_size': args.hidden,
        'seq_len': args.seq_len,
        'c2i': c2i,
        'i2c': i2c
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("[✓] Model metadata saved to %s", metadata_path)

    # 生成样本文本
    logger.info("\n--- Sample (temp=%.2f) ---", args.temp)
    generated_text = sample(model, args.sample_seed, c2i, i2c, vocab_size,
                            length=args.sample_length, temp=args.temp,
                            hidden_size=args.hidden, seq_len_for_sampling=args.seq_len)
    logger.info(generated_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        sys.exit(1)
