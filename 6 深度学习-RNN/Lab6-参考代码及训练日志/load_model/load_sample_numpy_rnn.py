import numpy as np
import pickle
import argparse
from rnn_scratch_framework import RNN, sample, set_seed, setup_logging

logger = setup_logging()  # 初始化日志


def load_model_and_sample(weights_path, metadata_path, seed_str, length, temp):
    # 1. 加载元数据
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    vocab_size = metadata['vocab_size']
    hidden_size = metadata['hidden_size']
    seq_len = metadata['seq_len']
    c2i = metadata['c2i']
    i2c = metadata['i2c']

    logger.info("Loaded metadata: vocab_size=%d, hidden_size=%d, seq_len=%d", vocab_size, hidden_size, seq_len)

    # 2. 初始化模型结构
    model = RNN(vocab_size, hidden_size)
    logger.info("Model structure initialized.")

    # 3. 加载权重
    loaded_weights = np.load(weights_path)
    model.Why = loaded_weights['Why']
    model.by = loaded_weights['by']
    model.cell.Wxh = loaded_weights['Wxh']
    model.cell.Whh = loaded_weights['Whh']
    model.cell.bh = loaded_weights['bh']
    logger.info("Model weights loaded from %s", weights_path)

    # 4. 设置随机种子 (可选，但为了可复现性推荐)
    # set_seed(42) # 您可以根据需要设置

    # 5. 使用加载的模型进行采样
    logger.info("\n--- Sampling with loaded model (temp=%.2f) ---", temp)
    generated_text = sample(model, seed_str, c2i, i2c, vocab_size,
                            length=length, temp=temp,
                            hidden_size=hidden_size, seq_len_for_sampling=seq_len)
    logger.info(generated_text)
    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NumPy RNN and sample text")
    parser.add_argument("--weights", default="numpy_rnn_model.npz", type=str, required=True, help="Path to model weights (.npz file)")
    parser.add_argument("--metadata", default="numpy_rnn_model_meta.pkl", type=str, required=True, help="Path to model metadata (.pkl file)")
    parser.add_argument("--seed_str", type=str, default="JULIET: ", help="Seed string for sampling")
    parser.add_argument("--length", type=int, default=200, help="Length of text to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    load_model_and_sample(args.weights, args.metadata, args.seed_str, args.length, args.temp)