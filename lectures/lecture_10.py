from execute_util import link, image, text
from collections import Counter
import re
import torch.nn as nn
import torch


def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)
    text("## 10: 跨语言迁移与多语言学习")
    assignment_2()
    XLM()
    Q_and_A()


def XLM():
    text("### 分布式假设（Distributional Hypothesis）")
    text("分布式假设是自然语言处理中的一个基本理论，提出了单词的语义由上下文语境来定义。")
    text("Word2Vec等词嵌入方法正是基于分布式假设，通过分析大量文本中单词的共现模式来学习单词的向量表示。")
    link(title="[Word2Vec, 2013]", url="https://arxiv.org/pdf/1301.3781")
    image("images/distributional_hypothesis.png", width=600)
    text("🤔：GPT属于哪种范式？ BERT属于哪种范式？")
    image("images/word2vec_eval.png", width=600)
    text("共有 8869 个语义问题和 10675 个句法问题")
    image("images/word2vec_res1.png", width=600)
    image("images/word2vec_res2.png", width=600)
    image("images/word2vec_res3.png", width=600)
    image("images/word2vec_res4.png", width=600)

    text("### 词级别跨语言迁移（Cross-lingual Transfer）")
    text("如何让两个语言分别训练的词嵌入空间对齐？")
    link(title="[MUSE, 2018]", url="https://arxiv.org/pdf/1710.04087")
    text(r"$$\min_{\mathbf{W} \in \mathbb{R}^{d \times d}} \frac{1}{n} \sum_{i=1}^n \ell\left(\mathbf{W} \mathbf{x}_i, \mathbf{y}_i\right)$$")
    text("其中，$\mathbf{W}$ 是一个线性变换矩阵，$\mathbf{x}_i$ 和 $\mathbf{y}_i$ 分别是源语言和目标语言中的词向量，$\ell$ 是一个损失函数（如欧氏距离或余弦相似度）。")
    image("images/muse4.png", width=600)
    text("存在闭式解，问题的关键在于如何构造X和Y")
    text("- 双语词典（Bilingual Dictionary）")
    text("- 高资源语言 ↔ 低资源语言")
    text("- 有工作发现仅数十个通用单词如阿拉伯数字、星期、月份等即可")

    image("images/muse2.png", width=600)

    image("images/muse1.png", width=600)

    image("images/muse3.png", width=600)

    image("images/muse_res1.png", width=400)
    image("images/muse_res2.png", width=600)
    text(r"$$\operatorname{CSLS}\left(W x_s, y_t\right)=2 \cos \left(W x_s, y_t\right)-r_{\mathrm{T}}\left(W x_s\right)-r_{\mathrm{S}}\left(y_t\right)$$")
    image("images/muse_res3.png", width=600)

    text("### 句子级别跨语言迁移（Cross-lingual Transfer）")

    image("images/unmt1.png", width=800)

    image("images/unmt2.png", width=400)

    image("images/unmt3.png", width=600)

    image("images/unmt4.png", width=600)


def assignment_2():
    text("### 作业2：训练自己的GPT模型 & 参加BabyLM挑战赛")

    text("- 数据集：多语言平行语料"), link(title="[massive]", url="https://huggingface.co/datasets/AmazonScience/massive")
    text("- **个人作业任务要求**：")
    text("1. 基于作业1训练的分词器构建微型GPT语言模型。")
    text("2. 在个人电脑上跑通训练流程")
    text("3. 对学习率、训练步数、预设词表大小等超参数进行调优，并分析其对训练的影响。")
    text("4. 实践报告**至少包含两个结论**，多多益善")
    text("<mark>**DDL：5月20日 24:00**</mark>")

    text("- **团队挑战赛**：")
    text("1. 以小组为单位参加BabyLM挑战赛（https://babylm.github.io/），使用BabyLM提供的10M预训练数据训练一个语言模型，并获得官方评测结果。")
    text("2. 本次竞赛可以探讨的方向包括但不限于：")
    text("-  关心语言：英文、中文、荷兰语的数据比例")
    text("-  关心数据：高效学习数据形式")
    text("-  关心架构：更好的分词器、更好的模型架构例如Qwen Gated Attention等"), link(title="[Qwen Gated Attention]", url="https://arxiv.org/pdf/2505.06708")
    text("-  关心训练：扩散语言模型等新型训练方式"), link(title="[LLaDA]", url="https://arxiv.org/abs/2502.09992")
    text("-  其他有兴趣的方向")
    text("自愿报名，组队成功后会提供相关资源和指导，与课程成绩无关，欢迎大家积极参与！")

    text("- **babylm数据质量分析**")
    text("- **Deepspeed分布式训练+Muon高效优化器**")
    text("零冗余优化（Zero Redundancy Optimizer，ZeRO）是一种分布式训练优化技术，旨在通过分散模型参数、梯度和优化器状态来减少内存占用，从而允许训练更大的模型。")
    image("images/zero.png", width=600)
    link(title="[Trainer+Deepspeed]", url="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.deepspeed")
    link(title="[GPU Kernel]", url="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.use_liger_kernel")
    link(title="[torch_compile]", url="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.torch_compile")

    text("Muon是一种高效的优化器，专为大规模分布式训练设计，能够在保持训练效率的同时显著降低内存使用。")
    link(title="[Muon Optimizer]", url="https://kexue.fm/archives/10592")
    link(title="[Deepspeed+Muon]", url="https://www.deepspeed.ai/docs/config-json/")
    link(title="[Kimi2.5、GLM5、DeepseekV4]", url="https://kexue.fm/archives/10739")

def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
