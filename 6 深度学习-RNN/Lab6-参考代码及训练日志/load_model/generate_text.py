import argparse
import os
import requests
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np


# 从训练代码中复制过来的模型定义
class CharLSTM(nn.Module):
    def __init__(self, vocab: int, embed_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out[:, -1, :])
        return logits, hidden


# 从训练代码中复制过来的数据构建函数
def download_shakespeare(path: str):
    if os.path.exists(path):
        return
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)


def build_dataset(text: str, seq_len: int):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c, len(chars)


# 从训练代码中复制过来的文本生成函数
def generate(model, seed: str, c2i, i2c, vocab: int, length=400, temp=1.0, device="cpu"):
    model.eval()
    out = seed
    seq = [c2i[c] for c in seed]
    hidden = None
    for _ in range(length):
        x = torch.tensor(seq[-args.seq_len:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, hidden = model(x, hidden)
        probs = torch.softmax(logits.squeeze() / temp, dim=0).cpu().numpy()
        idx = np.random.choice(vocab, p=probs)
        out += i2c[idx]
        seq.append(idx)
    return out


# 主函数
def main():
    global args
    parser = argparse.ArgumentParser(description="Generate text using a trained CharLSTM model.")
    parser.add_argument("--model_path", type=str, default="char_lstm.pth", help="Path to the trained model file")
    parser.add_argument("--data_path", type=str, default="shakespeare.txt", help="Path to the training data file")
    parser.add_argument("--seed", type=str, default="Juliet: ", help="Initial seed text for generation")
    parser.add_argument("--length", type=int, default=400, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--seq_len", type=int, default=120, help="Sequence length used during training")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    # 下载数据文件（如果尚未存在）
    download_shakespeare(args.data_path)

    # 读取数据并构建字符映射
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    c2i, i2c, vocab_size = build_dataset(text, args.seq_len)

    # 实例化模型并加载权重
    model = CharLSTM(
        vocab=vocab_size,
        embed_size=256,
        hidden=512,
        layers=3,
        dropout=0.3
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # 生成并打印文本
    generated_text = generate(
        model,
        seed=args.seed,
        c2i=c2i,
        i2c=i2c,
        vocab=vocab_size,
        length=args.length,
        temp=args.temperature,
        device=args.device
    )
    print("Generated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
