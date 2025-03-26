import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer
from torch.utils.data import TensorDataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from mask import create_masks
from tqdm import tqdm
import numpy as np
from transformers import FSMTTokenizer
import os


# 超参数设置（与论文一致）
SRC_VOCAB_SIZE = 58101  # 根据论文，WMT 2014 数据集使用了 37000 的源语言词汇表
TGT_VOCAB_SIZE = 58101  # 目标语言词汇表大小
D_MODEL = 512  # 模型维度 512
NUM_HEADS = 8  # 多头注意力头数
NUM_LAYERS = 6  # 编码器和解码器的层数
D_FF = 2048  # 前馈网络的隐藏层维度
MAX_SEQ_LENGTH = 128  # 最大序列长度  128
BATCH_SIZE = 64  # 每批次大小
EPOCHS = 1   # 训练周期数  10
LEARNING_RATE = 0.0001  # 学习率
WARMUP_STEPS = 4000  # 论文中的 warmup 步数



# 加载处理好的英文和德文索引数据
# en_idx 是英文句子的索引表示，de_idx 是德文句子的索引表示
en_idx = torch.load('./en_processed_test_indexes.pt')
de_idx = torch.load('./de_processed_test_indexes.pt')
# 为了提高测试速度，这里只选取前 100000 个样本进行测试
# 直接选取前 100000 个样本，可根据实际情况调整
en_idx = en_idx[:100000]
de_idx = de_idx[:100000]

# 加载预训练的分词器，用于将索引和词元进行转换
# 这里的路径指向预训练的分词器所在位置
tokenizer = FSMTTokenizer.from_pretrained('./wmt19-en-de')

# 创建测试数据集和数据加载器
# TensorDataset 将英文和德文索引数据组合成一个数据集
test_dataset = TensorDataset(en_idx, de_idx)
# DataLoader 用于批量加载数据，这里设置批量大小为 32，不进行数据打乱
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载 Transformer 模型
# TransformerConfig 是模型的配置类，用于初始化模型的参数
# 创建模型
model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=0.1)
# 检查是否有可用的 GPU，如果有则使用 GPU 进行计算，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到指定设备上
model = model.to(device)
# 加载预训练的模型权重
# 这里的路径指向保存的模型权重文件
model.load_state_dict(torch.load('./transformer_epoch_1_weights.pth'))
# 将模型设置为评估模式，关闭一些在训练时使用的特殊层（如 Dropout）
model.eval()

# 定义损失函数
# 使用交叉熵损失函数，忽略填充索引（这里填充索引为 1）
criterion = nn.CrossEntropyLoss(ignore_index=1)

# 添加平滑函数，用于计算 BLEU 分数时避免出现零分情况
smoother = SmoothingFunction()

def index_to_token(seq):
    """
    将索引序列转换为词元序列
    :param seq: 索引序列
    :return: 词元序列
    """
    return tokenizer.convert_ids_to_tokens(seq)

def get_seq_before_eos(seq):
    """
    获取 EOS（结束标记）之前的序列
    :param seq: 输入序列
    :return: EOS 之前的序列
    """
    try:
        # 查找 EOS 标记（索引为 2）的位置
        eos_index = seq.index(2)
        # 返回 EOS 之前的序列
        return seq[:eos_index]
    except ValueError:
        # 如果序列中没有 EOS 标记，则返回整个序列
        return seq

# 初始化两个列表，用于存储所有的预测结果和目标结果
all_preds = []
all_targets = []

# 禁用梯度计算，减少内存消耗并提高计算速度
with torch.no_grad():
    # 遍历测试数据加载器中的每个批次
    for src, tgt in tqdm(test_dataloader, desc="Testing"):
        # 将源序列（英文）和目标序列（德文）移动到指定设备上
        src = src.to(device)
        tgt = tgt.to(device)


        # 模型进行前向传播，得到预测结果
        output = model(src, tgt[:, :-1])

        # 获取预测结果，通过取最大值的索引得到每个位置的预测词元索引
        preds = output.argmax(dim=-1).cpu().numpy()
        # 获取目标序列的真实词元索引，去掉第一个开始标记
        targets = tgt[:, 1:].cpu().numpy()

        # 遍历每个批次中的每个样本
        for pred, target in zip(preds, targets):
            # 获取预测结果中 EOS 之前的序列
            pred_before_eos = get_seq_before_eos(pred.tolist())
            # 获取目标结果中 EOS 之前的序列
            target_before_eos = get_seq_before_eos(target.tolist())

            # 将索引序列转换为词元序列
            token_pred = index_to_token(pred_before_eos)
            token_target = index_to_token(target_before_eos)

            # 将预测结果和目标结果添加到对应的列表中
            all_preds.append(token_pred)
            all_targets.append([token_target])

# 计算 BLEU 分数
# corpus_bleu 用于计算整个语料库的 BLEU 分数
# smoother.method1 是平滑函数，weights 是不同 n-gram 的权重
bleu_score = corpus_bleu(all_targets, all_preds,
                         smoothing_function=smoother.method1,
                         weights=(0.25, 0.25, 0.25, 0.25))

# 打印 BLEU 分数，保留四位小数
print(f"BLEU Score: {bleu_score:.4f}")

# 打印翻译示例
print("\n翻译示例:")
# 只打印前 5 个翻译示例，如果样本数不足 5 个，则打印所有样本
for i in range(min(5, len(all_preds))):
    # 将词元序列转换为完整的句子
    target_sentence = tokenizer.convert_tokens_to_string(all_targets[i][0])
    pred_sentence = tokenizer.convert_tokens_to_string(all_preds[i])
    # 打印目标译文和模型译文
    print(f"\n目标译文: {target_sentence}")
    print(f"模型译文: {pred_sentence}")