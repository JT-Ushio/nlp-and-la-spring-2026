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


def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
