from execute_util import link, image, text
from collections import Counter
import re
import torch.nn as nn
import torch


def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)
    text("## 04: 语言模型与序列建模 (II)")
    assignment_1()
    what_is_lm()

    Q_and_A()

def assignment_1():
    text("### 作业1：BPE分词器的应用")
    text("- 数据集：多语言平行语料"), link(title="[massive]", url="https://huggingface.co/datasets/AmazonScience/massive")
    text("- 分词器构建：Huggingface Tokenizer"), link(title="[tokenizer demo]", url="https://huggingface.co/learn/llm-course/chapter2/4"), link(title="[tokenizer build]", url="https://huggingface.co/learn/llm-course/chapter6/8")
    text("- 任务要求：")
    text("1. 从massive数据集中选择**一组语言**（如英语、汉语、德语等），并使用Huggingface Tokenizer库训练一个**基于BPE算法的分词器**。")
    text("2. 提交**实践报告（飞书文档）**以及分词器的**词表文件（vocabulary.txt，merge.txt）**。")
    text("3. 实践报告**至少包含两个结论**，多多益善，可选以下分析角度")
    text("-  预设词表大小的影响")
    text("-  所选语言集合的影响（黏着/孤立/屈折是否均衡、汉藏/印欧/亚非语系是否均衡）")
    text("-  语言领域是否均衡的影响")
    text("-  BPE的分词结果分析（如自动分词是否对齐语素、分词序列平均长度等）")
    text("-  其他感兴趣的分析角度")
    text("4. <mark>**DDL：4月15日 24:00**</mark>")


def what_is_lm():

    text("### 什么是语言模型（Language Model）？")
    text("- 语言模型目标是建模自然语言的概率分布，是自然语言理解与自然语言生成任务的重要支撑。")
    text("- 大量的研究从n元语言模型和神经语言模型等不同角度开展了系列工作。")
    text("- 语言模型的核心问题是：给定一个字符串序列，计算它在真实语言分布中的概率。")
    text("- 给定一个字符串序列 $X=w_1 w_2 \cdots w_n$，语言模型的目标是计算 $P(X)=P(w_1, w_2, \ldots, w_n)$。")

    text("#### 链式法则（Chain Rule）")
    text("- 联合概率可以通过链式法则分解为条件概率的乘积：")
    text("- $P(w_{1:n}) = P(w_1) P(w_2|w_1) P(w_3|w_{1:2}) \cdots P(w_n|w_{1:n-1}) = \prod_{i=1}^n P(w_i|w_{1:i-1})$")
    text("- 通过链式法则，我们将联合概率分解为一系列条件概率的乘积，这些条件概率描述了每个词在给定前面所有词的情况下出现的概率。")
    text("- 例如，对于句子“我喜欢自然语言处理”，我们可以将其概率分解为：")
    text("- $P(我喜欢自然语言处理) = P(我) P(喜欢|我) P(自然|我喜欢) P(语言|我喜欢自然) P(处理|我喜欢自然语言)$")
    text("- 语言模型的训练目标是最大化训练语料中所有句子的联合概率，即最大化 $\prod_{X \in Corpus} P(X)$，或者等价地最小化负对数概率 $-\sum_{X \in Corpus} \log P(X)$。")

    text("#### 马尔可夫假设（Markov Assumption）")
    text("- 直接计算条件概率 $P(w_i|w_1, w_2, \ldots, w_{i-1})$ 仍然非常复杂，因为它依赖于前面所有的词。")
    text("- 马尔可夫假设简化了这个问题，假设每个词的出现只依赖于前面有限数量的词。")
    text(r"- 例如，二元语言模型（Bigram Language Model）假设 $P(w_i|w_1, w_2, \ldots, w_{i-1}) \approx P(w_i|w_{i-1})$，即每个词只依赖于前一个词。")
    text(r"- 三元语言模型（Trigram Language Model）假设 $P(w_i|w_1, w_2, \ldots, w_{i-1}) \approx P(w_i|w_{i-2}, w_{i-1})$，即每个词依赖于前两个词。")
    text("- 通过马尔可夫假设，我们将条件概率简化为依赖于有限上下文的概率，从而大大减少了模型需要估计的空间。")
    text("- 例如，在二元语言模型中，我们只需要估计 $|\mathcal{V}|^2$ 个参数，而在三元语言模型中，我们需要估计 $|\mathcal{V}|^3$ 个参数，这比原始的 $|\mathcal{V}|^n$ 要小得多。")
    text("- 但是，马尔可夫假设也带来了信息损失，因为它忽略了更长距离的依赖关系，这可能会影响模型的性能。")

    text("#### 基于梯度传播的最优化理论")
    text("**损失函数或目标函数：衡量模型预测值与真实值之间差异的函数。**")
    text("- 正确率（Accuracy）：预测正确的样本数占总样本数的比例。")
    text("- 准召率（Precision and Recall）：用于评估分类模型的性能，特别是在不平衡数据集上。")
    text("- 交叉熵损失（Cross-Entropy Loss）：常用于分类问题，衡量两个概率分布之间的差异。")
    text("- 铰链损失（Hinge Loss）：常用于支持向量机等模型，衡量预测值与真实值之间的差异。")
    text("- 排序损失（Ranking Loss）：常用于排序任务，衡量预测偏序与真实偏序之间的差异。")
    text("- ......")
    text("- NOTE: 损失函数依赖测试数据集的分布")

    text("**优化参数**")
    text("- 随机梯度下降（Stochastic Gradient Descent, SGD）：每次迭代使用一个样本或一个小批量样本来更新模型参数。")
    text("- 示例：初始参数为")
    text(r"$$\theta_1^{(0)} = 0, \quad \theta_2^{(0)} = 0$$")
    text(r"$$f\left(\theta_1, \theta_2\right)=\left(\theta_1-1\right)^2+2\left(\theta_2+2\right)^2$$")
    text("- 计算偏导：")
    text(r"$$\frac{\partial f}{\partial \theta_1} = 2(\theta_1 - 1), \quad \frac{\partial f}{\partial \theta_2} = 4(\theta_2 + 2)$$")
    text("- 更新参数：")
    text(r"$$\theta_1^{(t+1)} = \theta_1^{(t)} - \eta \cdot 2(\theta_1^{(t)} - 1), \quad \theta_2^{(t+1)} = \theta_2^{(t)} - \eta \cdot 4(\theta_2^{(t)} + 2)$$")
    image("images/sgd.png", width=600)

    text("🤔：如果目标是最大化怎么办？")

    text("#### 基于前馈神经网络的语言模型")
    text("- 神经网络语言模型的目标是根据历史单词对下一时刻词进行预测。")
    text("- 前馈神经网络是固定长度、静态的历史信息。")
    image("images/mlp_lm.png", width=600)


    class MLP_LM(nn.Module):
        def __init__(self, n_gram=3):
            super().__init__()

            self.vocab_size = 16
            self.embedding_size = 3
            self.n_gram = n_gram

            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
            self.hidden_size = 10

            self.mlp = nn.Sequential(
                nn.Linear(self.embedding_size * n_gram, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.vocab_size)
            )

        def forward(self, x_i_1, x_i_2, x_i_3, target=None):
            x_i_1 = self.embedding(x_i_1)  # (batch_size, embedding_size)
            x_i_2 = self.embedding(x_i_2)
            x_i_3 = self.embedding(x_i_3)

            x = torch.cat([x_i_1, x_i_2, x_i_3], dim=-1)  # (batch_size, embedding_size * 3)
            logits = self.mlp(x)  # (batch_size, vocab_size)

            loss = None
            if self.training and target is not None:
                loss = nn.CrossEntropyLoss()(logits, target)

            return logits, loss

    text("#### 基于循环神经网络的语言模型")
    text("- RNN建模完整的历史信息，语言模型的架构突破")
    text("- RNN是动态的、任意长度的历史信息。")
    text("- RNN的公式：")
    text("- RNN梯度消失/爆炸问题：求导时，链式法则导致梯度在长序列中逐渐变小，难以捕捉长距离依赖关系。")
    image("images/rnn_lm.png", width=600)


    class RNN_LM(nn.Module):
        def __init__(self, vocab_size, embedding_size, hidden_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(vocab_size, embedding_size)
            self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.rnn(x, hidden)
            out = self.fc(out)
            return out, hidden

    image("images/lstm_lm.png", width=600)
    text("- 缓解了梯度消失问题，能够捕捉更长距离的依赖关系。")
    text("- 门控网络：输入门（Input Gate）控制新信息的引入，遗忘门（Forget Gate）控制旧信息的遗忘，输出门（Output Gate）控制当前状态对输出的影响。")
    text("- 本质上是梯度的流动路径的设计，允许梯度在较长的时间步内保持稳定，从而捕捉长距离依赖关系。")

    text("#### 基于Transformer的语言模型")
    text("- Transformer建模完整的历史信息，语言模型的架构突破")
    text("- Transformer是动态的、任意长度的历史信息。")
    image("images/transformer_lm.png", width=600)
    text("- Transformer的公式：")
    text("- Multi-head Self-Attention机制：每个词与输入序列中的所有词进行交互，捕捉全局依赖关系。")
    text(r"$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$")
    text("$$Q, K, V = XW_Q, XW_K, XW_V$$")
    text("- 通过自注意力机制，Transformer能够捕捉输入序列中任意位置之间的依赖关系，克服了RNN在处理长序列时的梯度消失问题。")

    self_attn = nn.MultiheadAttention(embed_dim=4, num_heads=1, bias=False, batch_first=True)
    embeddings = nn.Embedding(num_embeddings=16, embedding_dim=4)
    embeddings_input = embeddings(torch.tensor([[1, 3, 2, 4, 5]]))  # (batch_size, seq_length, d_model)
    embeddings_input_size = embeddings_input.size()  # @inspect embeddings_input_size
    attn_output, attn_weights = self_attn(embeddings_input, embeddings_input, embeddings_input)
    attn_output_size = attn_output.size()  # @inspect attn_output_size

    text("- Multi-head：将输入分成多个子空间进行自注意力计算，增强模型的表达能力。")
    text("$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1, head_2, \\ldots, head_h)W^O$$")
    text("$$head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$")
    text("- 通过多头机制，Transformer能够在不同的子空间中捕捉输入序列中的不同类型的依赖关系，从而提升模型的性能。")
    text("- GQA（Grouped-Query Attention）：将查询分组，减少参数量，同时保持模型性能。(不要求掌握细节)")
    image("images/gqa.png", width=600)

    causal_vs_bidirectional(self_attn=self_attn, input=embeddings_input)

    text("🤔：大模型的记忆")
    text("- 参数记忆")
    text("- 上下文记忆")
    text("- 外挂记忆（如检索增强模型）")

    text("$$\\text{LayerNorm}(x) = \\frac{x - \mu}{\sigma} \odot \gamma + \\beta$$")
    text("- 层归一化（Layer Normalization）：对每个位置的输入进行归一化，稳定训练过程，促进模型收敛。")
    layer_norm = nn.LayerNorm(normalized_shape=4)
    layer_norm_attn_output = layer_norm(attn_output)  # (batch_size, seq_length, d_model)
    layer_norm_output_size = layer_norm_attn_output.size()  # @inspect layer_norm_output_size

    text("$$\\text{FFN}(x) = \\text{max}(0, xW_1 + b_1)W_2 + b_2$$")
    text("- 前馈神经网络（Feed-Forward Network, FFN）：每个位置的输出通过一个前馈神经网络进行非线性变换，增强模型的表达能力。")
    ffn = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4)
    )
    ffn_output = ffn(layer_norm_attn_output)  # (batch_size, seq_length, d_model)
    ffn_output_size = ffn_output.size()  # @inspect ffn_output_size

    text("- 残差连接（Residual Connection）：在每个子层（如自注意力层和前馈神经网络层）之后添加残差连接，帮助缓解深层网络中的梯度消失问题。")
    text("$$\\text{Residual}(x) = x + \\text{SubLayer}(x)$$")
    text("- 通过残差连接，Transformer能够训练更深的网络结构，捕捉更复杂的语言模式，从而提升语言模型的性能。")
    text("- 本质上是梯度的流动路径的设计，允许梯度在较深的网络中保持稳定")
    final_output = ffn_output + attn_output  # (batch_size, seq_length, d_model)
    final_output_size = final_output.size()  # @inspect final_output_size

    text(r"$$\text{Positional Encoding}(x) = x + \text{PE}$$")
    text("- 位置编码（Positional Encoding）：由于Transformer没有循环结构，需要通过位置编码为输入序列中的每个词提供位置信息，使模型能够捕捉序列中的顺序关系。")
    embeddings_input_2 = embeddings(torch.tensor([[1, 2, 3, 4, 5]]))  # (batch_size, seq_length, d_model)
    embeddings_input_2_size = embeddings_input_2.size()  # @inspect embeddings_input_size
    attn_output_2, _ = self_attn(embeddings_input_2, embeddings_input_2, embeddings_input_2)  # (batch_size, seq_length, d_model), (batch_size,
    attn_output_2_size = attn_output_2.size()  # @inspect attn_output_size
    is_same = torch.allclose(attn_output[:, -1, :], attn_output_2[:, -1, :])  # @inspect is_same

    text("- 绝对位置编码：为每个位置生成一个固定的编码，通常使用正弦和余弦函数；或者一个可学习的位置编码。")
    text("- 相对位置编码：根据词之间的相对位置生成编码，使模型能够更好地捕捉局部依赖关系。")
    text(r"$$\text{RoPE}(Q, K) = \text{Rotate}(Q, K)$$")
    text("- 提供位置信息，使模型能够区分不同位置的词，捕捉序列中的顺序关系。")
    text("🤔：单向注意力/因果（causal）注意力是否必须需要位置编码？")
    link(title="[NoPE]", url="https://arxiv.org/abs/2305.19466")

    text("Huggingface Transformers库中，语言模型的实现通常基于Transformer架构，并提供了丰富的预训练模型和工具，方便研究者和开发者进行语言建模任务。")
    text("- GPT in Huggingface Transformers。")
    image("images/llama_config.png", width=600)
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=256,              # d_model
        intermediate_size=1024,       # FFN dimension (4x hidden)
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,        # 保持与 attention heads 相同（不使用GQA）
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False
    )
    my_small_llama = LlamaForCausalLM(config)
    total_params = f"Number of parameters: {my_small_llama.num_parameters()/1e6:.2f}M" # @inspect total_params

    text("KV cache：在生成阶段，Transformer可以缓存之前计算的键（Key）和值（Value），避免重复计算，提高效率。")
    link(title="[KV Cache Demo]", url="https://www.dailydoseofds.com/p/kv-caching-in-llms-explained-visually/")

    text("**掩码语言模型**")
    text("- BERT in Huggingface Transformers。")
    text("- 掩码语言模型（Masked Language Model, MLM）在训练过程中随机掩盖输入序列中的一些词，并要求模型预测这些被掩盖的词。")
    text("- 例如，对于句子“我喜欢自然语言处理”，我们可以随机掩盖其中的一个词，如“我喜欢[掩码]语言处理”，模型的任务是预测被掩盖的词“自然”。")
    text("- 掩码语言模型通过引入掩码机制，使模型能够学习到更丰富的上下文信息，从而提升模型在自然语言理解任务中的性能。")

    text("🤔：掩码语言模型能做生成吗？")
    text("🤔：掩码语言模型与因果语言模型各有什么优劣？")

def causal_vs_bidirectional(self_attn, input):
    _, attn_weights_bidirectional = self_attn(input, input, input) # @inspect attn_weights_bidirectional
    attn_mask = torch.triu(torch.ones(input.size(-2), input.size(-2)), diagonal=1).bool() # @inspect attn_mask
    _, attn_weights_causal = self_attn(input, input, input, attn_mask=attn_mask, is_causal=True) # @inspect attn_weights_causal
    text("- 双向（Bidirectional）自注意力：每个位置的查询可以与输入序列中的所有位置进行交互，捕捉全局依赖关系。")
    text("- 因果（Causal）自注意力：每个位置的查询只能与输入序列中该位置之前的位置进行交互，确保模型在生成阶段只能利用历史信息，避免**信息泄露**。")


def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
