from execute_util import link, image, text
from collections import Counter
import re
import torch.nn as nn
import torch


def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)
    text("## 09: 有监督微调")
    how_to_train_lm()
    assignment_2()

    Q_and_A()


def how_to_train_lm():
    text("### 语言模型的预测目标：Next Token Prediction")
    image("images/next_token_prediction.png", width=400)
    text("预测标签是输入序列的移位，`-100` 表示不计算损失的部分。")

    image("images/next_token_prediction_with_eos1.png", width=400)
    text("需要增加一个特殊的结束符（EOS）来标识序列的结束，以便模型能够正确地学习到序列的边界。")

    image("images/next_token_prediction_with_eos2.png", width=400)
    text("可以手动把最后默认补的 `-100` 替换成 EOS 的 token id，这样模型就能学习到在序列结束时预测 EOS。")

    link(title="[Transformers CausalLMLoss]", url="https://github.com/huggingface/transformers/blob/f397b9e651dc7387de6bce551895619dfb1ec4f0/src/transformers/loss/loss_utils.py#L46")
    image("images/ntp_loss_func.png", width=400)

    text("### 指令遵循（Instruction Following）")
    text("**指令遵循（Instruction Following）**：训练语言模型根据给定的指令生成相应的输出。")
    image("images/instruction_following.png", width=600)
    text("指令遵循是语言模型微调的一个重要方向，尤其是在构建对话系统和生成式AI应用中。")
    text("- 第一部分是system prompt，提供模型的角色和行为指导；")
    text("- 第二部分是user prompt，提供具体的指令和输入；")
    text("- 第三部分是model response，模型根据指令和输入生成的输出。")
    text("- 每一部分需要用特殊标记分隔开来，以帮助模型理解不同部分的角色和内容。")

    image("images/instruction_following_multirun1.png", width=800)
    text("多轮对话/指令遵循，低效版。")
    text("- 每一轮都需要构造一条训练样本。")

    image("images/instruction_following_multirun2.png", width=800)
    text("多轮对话/指令遵循，高效版。")
    text("- 巧妙利用 `-100` 标签，避免重复计算上下文。")


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


def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
