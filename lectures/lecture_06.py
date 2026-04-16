from execute_util import link, image, text
from collections import Counter
import re
import torch.nn as nn
import torch


def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)
    text("## 05: 自然语言理解与生成")
    assignment_1()
    how_to_use_lm()
    how_to_train_lm()
    assignment_2()

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

    link(title="[Getting started with GitHub Copilot CLI]", url="https://docs.github.com/en/copilot/how-tos/copilot-cli/cli-getting-started")


def how_to_use_lm():
    text("### 如何使用语言模型？")
    text("- 预训练+微调（Pre-training + Fine-tuning）")
    image("images/BERT1.png", width=600)

    text("- 分类头（Classification Head）")
    image("images/BERT2.png", width=600)

    text("- 生成式预测（Generation Head）")
    image("images/T51.png", width=600)
    image("images/T52.png", width=600)

    text("- 参数学习 vs. 上下文学习（Parameter Learning vs. In-context Learning）")
    image("images/GPT1.png", width=600)


def how_to_train_lm():
    text("### 如何训练语言模型？")
    text("- 定义一个分词器")
    text("- 定义一个随机初始化的微型gpt2模型"), link(title="[transformers Models]", url="https://huggingface.co/docs/transformers/index")
    text("- 找到一个文本数据集")
    text("- 使用Trainer API训练模型")


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


def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
