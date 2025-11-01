"""
程序作者: GRZ
程序时间: 2025-11-01
程序目的: PyTorch实现RNN模型的基本框架: 实现文本分类

使用 IMDb 数据集，目标是预测一条电影评论是正面还是负面。
这是一个典型的序列数据问题，利用RNN（循环神经网络）来捕捉文本中的上下文信息
"""

# Import Library
import os
import logging
import logging.config
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import torch.nn as nn

## logging设置
### 载入配置文件
logging.config.fileConfig('config.ini')
### 获取logger
#### 输出到控制台
consolelogger = logging.getLogger('ConsoleOutput')
#### 输出到文件
filelogger = logging.getLogger()

# 忽略DEBUG级别的日志
logging.getLogger().setLevel(logging.INFO)

# ================================
# 全局参数设置
# ================================
max_len = 200  # 序列最大长度
batch_size = 32  # 批次大小

"""
	从 GloVe 文本文件加载词向量。
	返回: (word2idx, embeddings_matrix)
"""

def load_glove_embeddings(glove_path, embedding_dim=100):
	import numpy as np
	import torch

	word2idx = {}
	vectors = []

	try:
		with open(glove_path, 'r', encoding='utf8') as f:
			for idx, line in enumerate(f):
				line = line.strip()
				if not line:  # 跳过空行
					continue
					
				values = line.split()
				if len(values) < 2:  # 至少要有词和一个向量值
					filelogger.warning(f"第{idx+1}行格式错误: {line[:50]}...")
					continue
					
				word = values[0]
				
				# 尝试转换向量值，跳过无法转换的
				try:
					vector_values = []
					for val in values[1:]:
						# 过滤掉非数字字符
						if val.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
							vector_values.append(float(val))
					
					if len(vector_values) != embedding_dim:
						if idx < 10:  # 只打印前10个错误
							filelogger.warning(f"第{idx+1}行维度不匹配: 期望{embedding_dim}, 实际{len(vector_values)}")
						continue
					
					vector = np.asarray(vector_values, dtype=np.float32)
					word2idx[word] = len(vectors) + 1  # 从 1 开始编号，0 预留给 <PAD>
					vectors.append(vector)
					
				except (ValueError, TypeError) as e:
					if idx < 10:  # 只打印前10个错误
						filelogger.warning(f"第{idx+1}行数值转换错误: {str(e)}")
					continue

	except FileNotFoundError:
		filelogger.error(f"找不到词向量文件: {glove_path}")
		# 创建随机嵌入矩阵作为后备方案
		vocab_size = 10000  # 假设词汇表大小
		embeddings_matrix = torch.randn(vocab_size, embedding_dim)
		word2idx = {f'word_{i}': i for i in range(1, vocab_size)}
		return word2idx, embeddings_matrix

	if not vectors:
		filelogger.error("没有成功加载任何词向量")
		# 创建随机嵌入矩阵
		vocab_size = 10000
		embeddings_matrix = torch.randn(vocab_size, embedding_dim)
		word2idx = {f'word_{i}': i for i in range(1, vocab_size)}
		return word2idx, embeddings_matrix

	# 在最前面加一个 0 向量作为 <PAD>
	pad_vec = np.zeros((1, embedding_dim), dtype=np.float32)
	vectors = np.vstack([pad_vec, np.stack(vectors)])

	embeddings_matrix = torch.tensor(vectors)
	filelogger.info(f"成功加载 {len(word2idx)} 个词向量")

	return word2idx, embeddings_matrix

# ================================
# 文本清理与分词函数
# ================================
def tokenize(text):
	import re
	text = text.lower()
	text = re.sub(r"[^a-z0-9\s]", "", text)
	return text.split()


"""
	函数功能: 数据准备
	加载IMDb数据集并进行必要的预处理
"""
def read_data():

	# 设置随机种子
	torch.manual_seed(42)

	# 加载IMDb数据集
	dataset = load_dataset('imdb')
	train_data = dataset['train']
	test_data = dataset['test']
	
	filelogger.info(f'训练集样本数: {len(train_data)}')
	filelogger.info(f'测试集样本数: {len(test_data)}')

	# 加载GloVe词向量
	glove_path = 'wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt'  # GloVe词向量文件路径
	word2idx, embeddings_matrix = load_glove_embeddings(glove_path, embedding_dim=50)
	filelogger.info(f'词汇表大小: {len(word2idx)}')
	filelogger.info(f'嵌入矩阵形状: {embeddings_matrix.shape}')
	
	pad_idx = 0  # padding token 对应索引

	# ================================
	# 构建Dataset类
	# ================================
	class IMDBDataset(Dataset):
		def __init__(self, data, word2idx, max_len):
			self.data = data
			self.word2idx = word2idx
			self.max_len = max_len

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			text = self.data[idx]['text']
			label = self.data[idx]['label']

			tokens = tokenize(text)
			# 转索引（未知词→0）
			indices = [self.word2idx.get(tok, 0) for tok in tokens[:self.max_len]]
			# filelogger.info(f"indices的形状:{torch.tensor(indices, dtype=torch.long).shape}")
			return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

	# ================================
	# 动态批次填充函数
	# ================================
	def collate_fn(batch):
		# 将 batch 解包为两个元组：文本序列列表和对应的标签列表
		texts, labels = zip(*batch)
		# 对文本张量列表进行填充，使其在同一批次中具有相同长度
		padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
		# 将标签列表转换为 LongTensor 并与填充后的文本一起返回
		return padded_texts, torch.tensor(labels)

	# ================================
	# 构建 DataLoader
	# ================================
	# 构造训练集 Dataset 实例（将 IMDb 训练数据、词表和最大长度传入）
	train_dataset = IMDBDataset(train_data, word2idx, max_len)
	# 构造测试集 Dataset 实例（将 IMDb 测试数据、词表和最大长度传入）
	test_dataset = IMDBDataset(test_data, word2idx, max_len)

	# 为训练集创建 DataLoader：使用批量大小、打乱数据并指定 collate_fn 进行动态填充
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	# 为测试集创建 DataLoader：不打乱数据，使用相同的 collate_fn 进行动态填充
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	return train_loader, test_loader, embeddings_matrix, pad_idx

"""
	定义简单的RNN模型
"""
class RNNModel(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5, pad_idx=0, embeddings_matrix=None):
		"""
		input_dim: 词汇表大小
		embedding_dim: 词向量维度（例如 GloVe=50/100/300）
		hidden_dim: RNN 隐层维度
		output_dim: 输出维度（IMDb 二分类=2）
		n_layers: RNN 层数
		dropout: dropout 概率
		pad_idx: padding 索引
		embeddings_matrix: 预训练词向量矩阵（可选）
		"""
		super().__init__()

		# Embedding 层
		if embeddings_matrix is not None:
			self.embedding = nn.Embedding.from_pretrained(
				embeddings_matrix, freeze=False, padding_idx=pad_idx)
		else:
			self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

		# RNN 层
		self.rnn = nn.RNN(
			embedding_dim,
			hidden_dim,
			num_layers=n_layers,
			batch_first=True,
			dropout=dropout if n_layers > 1 else 0
		)

		# 全连接输出层
		self.fc = nn.Linear(hidden_dim, output_dim)

		# Dropout
		self.dropout = nn.Dropout(dropout)

	def forward(self, text, text_lengths=None):
		"""
		text: (batch_size, seq_len)
		text_lengths: 序列长度 (batch_size,) —— 用于 pack_padded_sequence
		"""
		# 1️⃣ embedding lookup
		embedded = self.embedding(text)  # (B, L, E)

		# 2️⃣ 处理可变长度（如果提供 text_lengths）
		if text_lengths is not None:
			packed_embedded = nn.utils.rnn.pack_padded_sequence(
				embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
			packed_output, hidden = self.rnn(packed_embedded)
		else:
			packed_output, hidden = self.rnn(embedded)

		# 3️⃣ 取最后一层隐藏状态 (num_layers, batch, hidden_dim)
		hidden = self.dropout(hidden[-1])  # (batch, hidden_dim)

		# 4️⃣ 分类层
		output = self.fc(hidden)  # (batch, output_dim)
		return output

# ================================
# 模型训练函数
# ================================
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0.0
		correct, total = 0, 0

		for texts, labels in train_loader:
			# 检查输入形状
			## 碰到第一个batch时打印形状
			# if epoch == 0:
			# 	filelogger.info(f"输入文本形状: {texts.shape}, 标签形状: {labels.shape}")
			texts, labels = texts.to(device), labels.to(device)

			# 清空梯度
			optimizer.zero_grad()

			# 前向传播
			outputs = model(texts)  # (batch_size, num_classes)

			# 计算损失
			loss = criterion(outputs, labels)

			# 反向传播
			loss.backward()
			optimizer.step()

			# 统计指标
			total_loss += loss.item()
			preds = outputs.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

		train_acc = correct / total
		filelogger.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")


# ================================
# 模型评估函数
# ================================
def evaluate_model(model, test_loader, device):
	model.eval()
	correct, total = 0, 0
	with torch.no_grad():
		for texts, labels in test_loader:
			texts, labels = texts.to(device), labels.to(device)
			outputs = model(texts)
			preds = outputs.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

	test_acc = correct / total
	filelogger.info(f"测试集准确率: {test_acc:.4f}")
	return test_acc


"""
	函数功能: 主程序入口
	输入: 
	输出: 
"""
if __name__ == '__main__':
	# ================================
	# 1️⃣ 数据准备
	# ================================
	train_loader, test_loader, embeddings_matrix, pad_idx = read_data()

	# 词表大小和参数设置
	vocab_size = embeddings_matrix.shape[0]
	embedding_dim = embeddings_matrix.shape[1]
	hidden_dim = 128
	output_dim = 2  # IMDb 为二分类任务
	n_layers = 1
	dropout = 0.5

	# ================================
	# 2️⃣ 模型初始化
	# ================================
	model = RNNModel(
		input_dim=vocab_size,
		embedding_dim=embedding_dim,
		hidden_dim=hidden_dim,
		output_dim=output_dim,
		n_layers=n_layers,
		dropout=dropout,
		pad_idx=pad_idx,
		embeddings_matrix=embeddings_matrix
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	# 3️⃣ 损失函数与优化器
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	# 4️⃣ 训练模型
	num_epochs = 3  # 训练轮数
	train_model(model, train_loader, criterion, optimizer, device, num_epochs)

	# 5️⃣ 测试模型
	test_acc = evaluate_model(model, test_loader, device)
	filelogger.info(f"最终测试集准确率: {test_acc:.4f}")




