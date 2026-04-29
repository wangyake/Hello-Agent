import torch 
import torch.nn as nn 
import math

# PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i+1/d_model))
# i 是维度索引(从0到d_model / 2)
# d_model 是模型维度，一般为 512

# 10000 的作用：控制位置编码的「波长范围」，
# 让不同维度的位置编码，波长从 2π 覆盖到 10000×2π，
# 既有短周期（细粒度位置），又有长周期（全局相对位置）

# 为什么选择 10000？
# 10000 → 波长覆盖范围刚好适合 NLP 任务（句子最长几千）
# 太小（如 1000）→ 长句子位置编码会重复，模型分不清位置
# 太大（如 10 万）→ 长波太长，浪费维度，对短句子没用
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout:float = 0.1, max_len:int=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # 创建一个足够长的位置编码矩阵，用于后续的加法操作
        # 为什么unsqueeze(1)? 因为position的维度是(max_len, 1)，而div_term的维度是(1, d_model)，所以需要广播到(max_len, d_model)才能进行加法
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # (d_model // 2, )
        
        # pe的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)

        # 偶数维度sin, 奇数维度cos
        pe[:, 0::2] = torch.sin(position.float() / div_term)
        pe[:, 1::2] = torch.cos(position.float() / div_term)

        # 注册pe为缓冲区，这样不会被视为模型参数，但会随模型移动和加载（如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是序列长度
        # x.size(2) 是模型维度
        # 输出维度(batch_size, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)]) # (batch_size, seq_len, d_model) === （:, :x.size(1)， ：）末尾维度广播

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V 线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算 QK^T
        # 为什么Q.dtype? 因为Q, K, V的dtype是float32，而math.sqrt(self.d_k)是float64，所以需要转换为float32
        # dk = torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype))
        # Q,K,V维度(batch_size, num_heads, seq_len, d_k)，K转置后维度(batch_size, num_heads, d_k, seq_len)
        # 输出维度(batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        # 维度(batch_size, num_heads, seq_len, seq_len)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 加权求和
        # 输出维度(batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn_probs, V)
        return output

    def split_head(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_head(self, x):
        batch_size, seq_len, num_heads, d_k = x.size()
        return x.transpose(1, 2).view(batch_size, seq_len, num_heads * d_k)

    def forward(self, query, key, value, mask):
        Q = self.split_head(self.W_q(query))
        K = self.split_head(self.W_k(key))
        V = self.split_head(self.W_v(value))
        
        # Q,K,V 此时维度 (batch_size, num_heads, seq_len, d_k)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_head(attn_output)
        output = self.W_o(output)
        return output

# 注意力曾是从整个序列中“动态地聚合”相关信息，FFN就是从这些信息中提取出更高级别的特征
# 关键在于“逐位置”，独立作用于序列中每一个词元向量（seq_len次）。
# 所有位置共享同一组网络权重，这种设计既保持了对每个位置进行独立加工的能力，又大大减少了模型的参数量。
# 网络结构非常简单：只需要两个线性变换，一个ReLU。
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # 位置级全连接层
    def forward(self, x):
        # x维度(batch_size, seq_len, d_model),这是多头注意力concat后的维度
        x = self.W_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W_2(x)
        # 输出维度(batch_size, seq_len, d_model)
        return x

# 编码器核心层：
class EncoderLayer(nn.Module):
    # params:
    # d_model: 模型维度
    # n_heads: 多头注意力头数
    # d_ff: 前馈层维度
    # dropout: dropout概率
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2.前馈层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# 解码器核心层：
class DecoderLayer(nn.Module):
    # params:
    # d_model: 模型维度
    # n_heads: 多头注意力头数
    # d_ff: 前馈层维度
    # dropout: dropout概率
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1.掩码多头自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 交叉注意力
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x