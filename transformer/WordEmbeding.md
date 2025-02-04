### **一、输入处理：从文本到向量**
#### **1.1 Tokenization（分词）**
假设输入句子是："The cat sat on the mat."

**步骤1：子词切分（以BERT为例）**
- 使用WordPiece分词器：
  - "The" → ["the"]（实际会转为小写）
  - "cat" → ["cat"]
  - "sat" → ["sat"]
  - "on" → ["on"]
  - "the" → ["the"]
  - "mat" → ["mat"]
- 添加特殊Token：
  - 开头加[CLS]，结尾加[SEP] → ["[CLS]", "the", "cat", "sat", "on", "the", "mat", "[SEP]"]
  
**词表映射**：每个Token转换为ID（例如[CLS]=101, "the"=1996, "cat"=5431等）

#### **1.2 构建输入矩阵**
假设：
- 词表大小V=30,000
- 嵌入维度d=768
- 最大序列长度L=512

每个Token ID通过**嵌入矩阵**转换为向量：
```python
# 嵌入矩阵形状：(V, d) = (30000, 768)
embedding_matrix = nn.Embedding(num_embeddings=V, embedding_dim=d) 
token_embeddings = embedding_matrix(input_ids) # shape=(batch_size, L, d)
```

---

### **二、位置编码（Positional Encoding）**
#### **2.1 位置编码公式**
**原始Transformer的位置编码**：
$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$
- `pos`：词语在序列中的位置（0 ≤ pos < L）
- `i`：维度索引（0 ≤ i < d/2）

**示例计算**（d=4, pos=1）：
```
PE(1,0) = sin(1 / 10000^(0/4)) = sin(1)
PE(1,1) = cos(1 / 10000^(0/4)) = cos(1)
PE(1,2) = sin(1 / 10000^(2/4)) = sin(1/100)
PE(1,3) = cos(1 / 10000^(2/4)) = cos(1/100)
```

#### **2.2 位置编码实现细节**
**代码实现**（PyTorch）：
```python
def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)          # shape=(seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)           # 偶数列用sin
    pe[:, 1::2] = torch.cos(position * div_term)           # 奇数列用cos
    return pe  # shape=(seq_len, d_model)
```

**关键设计原因**：
1. **正弦/余弦交替**：允许模型学习相对位置关系（因为`sin(a+b)`和`cos(a+b)`可用sin(a)和cos(a)线性表示）。
2. **指数衰减频率**：不同维度对应不同波长（从2π到20000π），捕捉多尺度位置信息。
3. **确定性生成**：无需训练，直接预计算。

---

### **三、嵌入层整合**
将Token嵌入和位置编码相加：
```python
final_embeddings = token_embeddings + positional_embeddings
```
**维度对齐**：两个张量必须都是(batch_size, L, d)

**为什么用加法而不是拼接**？
- 拼接会增加维度（d→2d），而加法保持维度不变，减少计算量。
- 实验证明加法足以让模型分离语义和位置信息。

---

### **四、层归一化（Layer Normalization）**
在Transformer的输入前先做层归一化（某些实现中在残差连接后做）：
```python
ln = nn.LayerNorm(d)  # 可学习参数：gamma和beta（形状为d）
norm_embeddings = ln(final_embeddings)
```

**归一化公式**：
$$
\text{Output} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
- `μ`：均值（沿d维度计算）
- `σ`：标准差
- `ε`：防止除零的小数（如1e-5）
- `γ`和`β`：可学习的缩放和平移参数

**作用**：
- 稳定训练：防止梯度爆炸/消失
- 加速收敛：使输入分布保持稳定

---

### **五、参数共享技巧**
#### **5.1 输入输出权重绑定（Weight Tying）**
在语言模型中，输入嵌入矩阵和输出层共享权重：
```python
# 输入嵌入矩阵：shape=(V, d)
embedding_matrix = nn.Embedding(V, d)
# 输出层：shape=(d, V)
output_layer = nn.Linear(d, V)
output_layer.weight = embedding_matrix.weight  # 共享权重
```

**数学原理**：
- 输入时，词向量是`E = embedding_matrix(w)`
- 输出时，logits计算为`E * W^T + b`（其中`W = embedding_matrix.weight`）
- 共享权重减少了参数量（从2Vd到Vd + d）

#### **5.2 缩放因子（Scale Factor）**
在部分实现中，对嵌入结果乘以√d：
```python
token_embeddings = token_embeddings * math.sqrt(d)
```
**原因**：在后续的注意力计算中，点积`QK^T`的值会随d增大而增长，导致softmax梯度消失。提前缩放可缓解此问题。

---

### **六、完整前向传播流程**
假设输入序列长度为L=5，d=768，batch_size=32：

1. **输入ID转换**：
   - 输入形状：(32, 5) ← 32个样本，每个样本5个Token
   - 查表得到Token嵌入：(32, 5, 768)

2. **加入位置编码**：
   - 位置编码形状：(5, 768)
   - 广播相加：(32, 5, 768) + (5, 768) → (32, 5, 768)

3. **层归一化**：
   - 计算每个Token的均值和方差（沿768维）
   - 输出形状不变：(32, 5, 768)

4. **Dropout**（可选）：
   - 训练时随机置零部分元素，防止过拟合

5. **输入到Transformer编码器**：
   - 作为第一层的输入

---

### **七、关键超参数影响**
#### **7.1 嵌入维度d的选择**
- **太小**（如d=128）：模型容量不足，无法捕捉复杂语义
- **太大**（如d=2048）：计算量平方级增长，可能过拟合
- **平衡点**：BERT-base用768，Large用1024，GPT-3用12288

#### **7.2 位置编码的替代方案**
- **可学习的位置嵌入**（如BERT）：
  ```python
  position_embeddings = nn.Embedding(L, d)  # 训练时更新
  ```
  - 优点：更灵活
  - 缺点：无法处理超过L的序列长度

- **相对位置编码**（如Transformer-XL）：
  - 编码相对距离而非绝对位置
  - 公式：$e_{ij} = \sin((i-j)/10000^{2k/d})$ 或可学习的

---

### **八、可视化理解**
#### **Token嵌入空间**
- 使用t-SNE降维后，相似词聚集：
  - "dog"、"puppy"、"canine"在相邻区域
  - "run"、"running"、"ran"形成动词簇

#### **位置编码波形**
- 不同位置的编码向量在低频和高频维度呈现独特波形组合
- 位置0和位置1的编码差异远大于位置100和101的差异（适应相对位置）

---

### **九、常见问题与解决方案**
#### **9.1 长序列处理**
- **问题**：位置编码预定义长度L=512，如何处理更长文本？
- **方案**：
  - 截断（BERT原始方法）
  - 外推（如ALiBi位置编码允许外推）
  - 分块处理（如Transformer-XL的循环机制）

#### **9.2 嵌入初始化**
- **标准方法**：用正态分布初始化（均值为0，方差为0.02）
- **特殊技巧**：在嵌入矩阵的底部（对应高频词）设置更小的初始值

#### **9.3 多语言嵌入**
- **挑战**：不同语言共享同一向量空间
- **解决方案**：
  - 使用共享子词词表（如XLM-R）
  - 对齐单语嵌入空间（通过翻译对）

---

通过以上细节，你应该对Transformer的词嵌入机制有了透彻理解。如果想验证这些知识，可以尝试以下实践：
1. 用PyTorch从头实现一个可训练的位置编码层
2. 可视化不同位置编码的波形差异
3. 对比使用/不使用层归一化时的训练速度差异