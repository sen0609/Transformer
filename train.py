import os
from torch.utils.data import DataLoader,TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from model import Transformer  # 假设 Transformer 定义在 model.py 文件中
from data_processing import load_and_process_data  # 假设数据处理在 data_processing.py 中
from transformers import MarianTokenizer

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

# 定义学习率调度器
class TransformerLRScheduler:
    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0  # 训练步数

    def step(self, optimizer):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 绘制损失曲线
def plot_loss(steps, losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=1.5, color='blue')
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss every 200 batches', fontsize=16)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(save_path)
    plt.close()

# 训练函数
def train():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预处理后的索引数据
    en_idx = torch.load('./en_processed_indexes.pt')
    de_idx = torch.load('./de_processed_indexes.pt')

    # 创建数据集和数据加载器
    dataset = TensorDataset(en_idx, de_idx)  # 将英文和德文数据配对
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  # 每个epoch前打乱数据
    )

    # 创建模型
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=0.1
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # 忽略填充符
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

    # 用于记录损失
    losses = []
    steps = []
    step_count = 0

    # 用于每200个batch记录日志
    batch_log_data = []
    epoch_log_data = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for src_batch, tgt_batch in batch_iterator:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()

            output = model(src_batch, tgt_batch[:, :-1])
            
            # 计算损失
            loss = criterion(output.view(-1, TGT_VOCAB_SIZE), tgt_batch[:, 1:].contiguous().view(-1))
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            current_lr = scheduler.step(optimizer)
            train_loss += loss.item()

            batch_iterator.set_postfix(loss=loss.item(), lr=current_lr)
            step_count += 1
            if step_count % 200 == 0:
                losses.append(loss.item())
                steps.append(step_count)
                plot_loss(steps, losses, 'transformer_loss.png')

                # 记录每200个batch的训练日志
                batch_log_data.append({
                    'Epoch': epoch + 1,
                    'Batch': step_count,
                    'Loss': loss.item(),
                    'Learning Rate': current_lr
                })

        epoch_loss = train_loss / len(dataloader)
        print(f'🚀 Epoch {epoch + 1}/{EPOCHS} Loss: {epoch_loss:.4f}')

        # 每个epoch结束后保存模型
        save_path = f"transformer_epoch_{epoch + 1}_weights.pth"
        torch.save(model.state_dict(), save_path)
        print(f"🚀 权重已保存: {save_path}")

        # 记录每个epoch的训练日志
        epoch_log_data.append({
            'Epoch': epoch + 1,
            'Epoch Loss': epoch_loss,
            'Final Learning Rate': current_lr
        })

        # 保存 epoch 日志
        epoch_csv_save_path = 'epoch_training_logs.csv'
        os.makedirs(os.path.dirname(epoch_csv_save_path), exist_ok=True)
        with open(epoch_csv_save_path, 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Epoch Loss', 'Final Learning Rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for log in epoch_log_data:
                writer.writerow(log)

    # 保存每200个batch的训练日志
    batch_csv_save_path = 'batch_training_logs.csv'
    os.makedirs(os.path.dirname(batch_csv_save_path), exist_ok=True)
    with open(batch_csv_save_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Batch', 'Loss', 'Learning Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for log in batch_log_data:
            writer.writerow(log)
    print(f"每200个batch训练日志已保存到 {batch_csv_save_path}")


# 启动训练
if __name__ == "__main__":
    train()
