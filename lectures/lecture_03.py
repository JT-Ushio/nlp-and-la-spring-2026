from execute_util import link, image, text
from collections import Counter
import re
import torch.nn as nn
import torch


def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)
    text("## 03: 语言模型与序列建模 (I)")
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
    def shuffle_string(s):
        import random
        char_list = list(s)
        random.shuffle(char_list)
        return ''.join(char_list)

    string = "语言模型目标是建模自然语言的概率分布，是自然语言理解与自然语言生成任务的重要支撑。" # @inspect string
    shuffled = shuffle_string(string) # @inspect shuffled
    text("$P(string) > P(shuffled)$")
    text("- 给定一个字符串序列 $X=w_1 w_2 \cdots w_n$，语言模型的目标是计算 $P(X)=P(w_1, w_2, \ldots, w_n)$。")
    text("- 对于任意词串$w_1, w_2, \ldots, w_n$，则有 $P(w_1, w_2, \ldots, w_n) \ge 0$")
    text("- 对于所有词串，则有 $\sum_{w_1, w_2, \ldots, w_n} P(w_1, w_2, \ldots, w_n) = 1$。")
    text("- 联合概率的计算空间是指数级别，如果把$w_1, w_2, \ldots, w_n$看作一个变量，那么它具有 $|\mathcal{V}|^n$ 种可能。")
    text("- 按照《现代汉语词典（第七版）》包含7万词条，句子长度按照20个词计算，模型参数量达到 $7.98*10^{96}$ 的天文数字。")

    text("#### 链式法则（Chain Rule）")
    text("- 联合概率可以通过链式法则分解为条件概率的乘积：")
    text("- $P(w_{1:n}) = P(w_1) P(w_2|w_1) P(w_3|w_{1:2}) \cdots P(w_n|w_{1:n-1}) = \prod_{i=1}^n P(w_i|w_{1:i-1})$")
    text("- 通过链式法则，我们将联合概率分解为一系列条件概率的乘积，这些条件概率描述了每个词在给定前面所有词的情况下出现的概率。")
    text("- 例如，对于句子“我喜欢自然语言处理”，我们可以将其概率分解为：")
    text("- $P(我喜欢自然语言处理) = P(我) P(喜欢|我) P(自然|我喜欢) P(语言|我喜欢自然) P(处理|我喜欢自然语言)$")
    text("- 语言模型的训练目标是最大化训练语料中所有句子的联合概率，即最大化 $\prod_{X \in Corpus} P(X)$，或者等价地最小化负对数概率 $-\sum_{X \in Corpus} \log P(X)$。")
    text("- 负对数似然（Negative Log-Likelihood, NLL）越小，模型对训练数据的拟合程度越好。")
    text("- NLL可以把条件概率相乘转化为条件概率的对数相加，计算更稳定，因此更常用。")
    text(r"- 困惑度（Perplexity）是语言模型评估的常用指标，定义为 $PPL(X) = 2^{-\frac{1}{N} \sum_{i=1}^N \log P(w_i|w_1, w_2, \ldots, w_{i-1})}$，其中 $N$ 是句子长度。困惑度越小，模型对数据的预测能力越强。")
    text(r"- 负对数似然和困惑度的关系：$PPL(X) = 2^{\frac{1}{N} NLL(X)}$，因此困惑度是负对数似然的指数函数。")
    text("- 语言模型的训练过程虽然采用的有监督方法，但是由于训练目标可以通过原始文本直接获得，从而使得模型的训练仅需要大规模无标注文本即可。语言模型也成为了典型的自监督学习（Self-supervised Learning）任务。")

    text("#### 马尔可夫假设（Markov Assumption）")
    text("- 直接计算条件概率 $P(w_i|w_1, w_2, \ldots, w_{i-1})$ 仍然非常复杂，因为它依赖于前面所有的词。")
    text("- 马尔可夫假设简化了这个问题，假设每个词的出现只依赖于前面有限数量的词。")
    text(r"- 例如，二元语言模型（Bigram Language Model）假设 $P(w_i|w_1, w_2, \ldots, w_{i-1}) \approx P(w_i|w_{i-1})$，即每个词只依赖于前一个词。")
    text(r"- 三元语言模型（Trigram Language Model）假设 $P(w_i|w_1, w_2, \ldots, w_{i-1}) \approx P(w_i|w_{i-2}, w_{i-1})$，即每个词依赖于前两个词。")
    text("- 通过马尔可夫假设，我们将条件概率简化为依赖于有限上下文的概率，从而大大减少了模型需要估计的空间。")
    text("- 例如，在二元语言模型中，我们只需要估计 $|\mathcal{V}|^2$ 个参数，而在三元语言模型中，我们需要估计 $|\mathcal{V}|^3$ 个参数，这比原始的 $|\mathcal{V}|^n$ 要小得多。")
    text("- 但是，马尔可夫假设也带来了信息损失，因为它忽略了更长距离的依赖关系，这可能会影响模型的性能。")

    text("🤔：语言模型能否反向建模？")
    text("- 反向语言模型（Backward Language Model）是指从句子末尾开始建模，计算 $P(w_n, w_{n-1}, \ldots, w_1)$。")
    text("🤔：语言模型是否对历史信息的顺序敏感？")
    string = "语言模型目标是建模自然语言的概率分布，在自然语言处理研究中具有重要的作用，是机器翻译、语音识别、输入法、句法分析等任务的支撑。语言模型是自然语言处理基础任务之一，大量的研究从n元语言模型和神经语言模型等不同角度开展了系列工作。" # @inspect string
    shuffled = shuffle_string(string) # @inspect shuffled
    text("✍️：打乱多少顺序信息会破坏理解？")


    text("#### 基于词频的语言模型")
    text("- 最简单的语言模型是基于词频的模型")
    text("- 例如 n 元语言模型可以通过统计训练语料得到 $P(w_i|w_{1:i-1})$")
    text(r"- $P(w_i|w_{1:i-1}) = \frac{C(w_{1:i-1}, w_i)}{C(w_{1:i-1})}$，其中 $C(\cdot)$ 表示词频.")
    text("- 可能存在数据稀疏问题，即许多词串在训练语料中未出现过，导致无法估计其概率。")
    text("- 需要超大规模语料库或平滑技术（Smoothing）。")
    text("- 平滑处理的基本思想是提高低概率，降低高概率，使整体的概率分布趋于均匀。")


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

def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
