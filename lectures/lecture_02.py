from execute_util import link, image, text
from collections import Counter
import re


def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)
    text("## 02: 词元与分词算法")

    what_is_tokens()
    how_to_tokenize()
    todo()

    Q_and_A()

def what_is_tokens():
    text("### 什么是词元（Token）？")
    text("- NLP中，词元（Token）是文本处理的基本单位。它可以是单词、子词或字符，取决于分词方法。")
    text("- 一个句子（Sentence）是由若干个词元组成的线性序列。")
    text(r"- $\mathbf{x} = (x_1, x_2, \ldots, x_n), ~~~ x_i \in \mathcal{V}$")
    text("- 每个词元 $x_i$ 都属于词表 $\mathcal{V}$。通常词表是静态的，包含了模型在**训练时见过**的所有词元。")
    text("- 词表大小（Vocabulary Size）是词表中词元的数量，通常在几万到几十万之间。")

    text("### 词元的粒度：单词、子词、字符")
    text("- 不同的分词粒度影响序列长度$n$和词表大小$|\mathcal{V}|$。")

    text("| 粒度 | 示例 | 序列长度 | 词表大小 |\n| :--- | :--- | :---: | :---: |\n| 单词（Word) | Hello world! → [Hello, world, !] | **较短** | **极大** |\n| 子词（Subword） | Hello world! → [He, llo, world, !] | **适中** | **适中** |\n| 字符（Character） | Hello world! → [H, e, l, l, o,  , w, o, r, l, d, !] | **极长** | **较小** |\n")
    text("#### 语言类型也会影响实际的词元粒度选择")
    text("- 屈折语：如拉丁语、俄语等，词内有表示语法意义的屈折词缀（如格、数、性）。")
    text("- 孤立语：如汉语、泰语等，词本身没有形态变化。")
    text("- 黏着语：如土耳其语、日语等，词由多个语素黏着而成。")

    text("举例，新华词典：其中单字条目约15200条，多字条目约32000条。")
    image("images/cidian.jpg", width=160)
    text("举例，德语的三种粒度统计。")
    image("images/german_tokens.png", width=600)

    text("| 粒度 | 优点 | 缺点 |\n| :--- | :--- | :---: |\n| 单词（Word) | 语义直观；计算量小 | 词表巨大；未登录词问题严重 |\n| 子词（Subword） | 兼顾计算量与语义；几乎消除OOV | 子词定义不明确 |\n| 字符（Character） | 词表最小；天然无OOV | 语义不直观；计算量大 |\n")

    text("- 子词定义不明确")
    image("images/subwords.png", width=300)
    text("- 未登录词（Out-of-Vocabulary, OOV）：指在训练词表中没有出现过的词元。")
    text("- 语义直观：")
    text("- Abwasserbehandlungsanlage@德语：")
    text("| 词素             | 含义           |\n| -------------- | ------------ |\n| **Abwasser**          | 废水 / 污水      |\n| **Behandlung** | 处理           |\n| **Anlage**     | 设施 / 装置 / 工厂 |\n")

    text("#### 特殊词元（special tokens）")
    text("- **[UNK]**: 未登录词（Out-of-Vocabulary）的占位符")
    text("- **[PAD]**: 填充词元（Padding Token），用于对齐序列长度")
    text("- **[CLS]**: 分类词元（Class Token），用于分类任务")
    text("- **[SEP]**: 分隔词元（Separator Token），用于分隔不同文本片段")
    text("- **[MASK]**: 掩码词元（Mask Token），用于掩码语言模型")
    text("- **[BOS]**: 序列开始词元（Beginning of Sequence Token）")
    text("- **[EOS]**: 序列结束词元（End of Sequence Token）")
    text("- ...任意自定义...")

    text("- Hello world!🌏你好迣鎅 → [BOS], He, llo, world, !, [UNK], 你, 好, [UNK], [UNK], [EOS]")

def how_to_tokenize():
    text("### 分词器")
    text("分词器用于在文本字符串与整数序列（Token）之间进行转换")
    image("images/tokenized-example.png", width=600)
    text("直观理解：把字符串拆分成常见的子片段")

    text("字符 → 整数")
    text("- 直接进行字母表映射（如ASCII码、单字表），❌ 中文上万 vs. 英文数百。")
    text("- 使用万国码（Unicode）字节进行映射。")
    A_ascii, A_utf_8 = "A".encode("ascii").hex(), "A".encode("utf-8").hex() # @inspect A_ascii, @inspect A_utf_8

    hello_en, hello_zh = "Hello".encode("utf-8").hex(), "你好".encode("utf-8").hex() # @inspect hello_en, @inspect hello_zh

    a5 = "¥".encode('utf-8').hex() # @inspect a5

    hand1 = "把".encode('utf-8').hex() # @inspect hand1
    hand2 = "提".encode('utf-8').hex() # @inspect hand2

    link(title="[CL202509] Tokenization Changes Meaning in Large Language Models: Evidence from Chinese", url="https://direct.mit.edu/coli/article/51/3/785/128327/Tokenization-Changes-Meaning-in-Large-Language")

    image("images/chinese_bpe.png", width=600)
    image("images/chinese_bpe_exp.png", width=600)
    image("images/chinese_bpe_res.png", width=600)

    text("🤔：当前分词器的发展趋势是否正确？→ 词表越大越智能吗？")
    text("🤔：如果问两个字的发音是否相同？")

    max_match_tokenizer()
    bpe_tokenizer()

    text("🤔：大模型能否精确知道字数？")


def bpe_tokenizer():
    text("#### BPE分词算法（Byte Pair Encoding Tokenizer）")
    text("- BPE是一种基于统计的分词方法，通过迭代合并最常见的字符对来构建词表。")
    image("images/bpe_training_algo.png", width=600)
    image("images/bpe_tokenize_algo.png", width=600)

    def get_stats(vocab):
        pairs = Counter() # @inspect pairs
        for word, freq in vocab.items():
            symbols = word.split() # @inspect symbols
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq # @inspect pairs
        return pairs

    def merge_vocab(pair, vocab):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {} # @inspect new_vocab, @inspect vocab
        for word in vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word] # @inspect new_vocab
        return new_vocab

    def bpe_training(corpus, num_merges=10):
        all_text = "Ġ".join(corpus)
        all_text = all_text.replace(" ", "Ġ")

        vocab = {}
        for word in all_text.split('Ġ'):
            if word:
                chars = ' '.join(list(word)) + ' </w>'
                vocab[chars] = vocab.get(chars, 0) + 1  # @inspect vocab

        merges = []

        for i in range(num_merges):
            pairs = get_stats(vocab) # @inspect pairs
            if not pairs:
                break
            best = max(pairs, key=pairs.get) # @inspect best
            vocab = merge_vocab(best, vocab) # @inspect vocab
            merges.append(best) # @inspect merges
            print(f"Merge {i+1}: {best} -> {''.join(best)} (freq: {pairs[best]})")

        vocabulary = Counter()
        for word, freq in vocab.items():
            tokens = word.split()
            for token in tokens:
                vocabulary[token] += freq # @inspect vocabulary

        return vocabulary, merges

    def bpe_tokenize(vocabulary, text, merges):
        text = text.replace(" ", "Ġ")
        tokens = [] # @inspect tokens
        for word in text.split('Ġ'):
            if not word:
                continue
            word_tokens = list(word) + ['</w>'] # @inspect word_tokens

            for merge in merges:
                new_tokens = [] # @inspect new_tokens
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == merge:
                        new_tokens.append(''.join(merge)) # @inspect new_tokens
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i]) # @inspect new_tokens
                        i += 1
                word_tokens = new_tokens

            tokens.extend(word_tokens) # @inspect tokens

        return tokens


    corpus = [
        "学自然语言处理",
        "我也喜欢语言学",
        "喜欢自然",
    ]

    vocabulary, merges = bpe_training(corpus, num_merges=4) # @inspect vocabulary, @inspect merges

    sentence = "语言学处理自然语言" # @inspect sentence
    tokens = bpe_tokenize(vocabulary, sentence, merges) # @inspect tokens

    image("images/bpe_tokenize_ex.png", width=600)


def max_match_tokenizer():
    text("#### 最大匹配分词算法（Maximum Matching Tokenizer）")
    text("- 又称贪心分词算法（Greedy Tokenizer），是一种基于词表的简单分词方法。")
    text("- 基本思想：从文本的开始位置出发，尝试匹配最长的词元，如果匹配成功则将该词元作为一个分词结果，并继续从下一个位置进行匹配；如果匹配失败，则将当前位置的单个字符作为一个词元，并继续匹配下一个位置。")
    image("images/max_match_algo.png", width=600)

    def max_match_tokenize(vocabulary, text):
        vocab = set(vocabulary)
        max_len = max(len(word) for word in vocab) if vocab else 0

        result = [] # @inspect result
        i = 0   # @inspect i

        while i < len(text):
            matched = False
            for j in range(min(max_len, len(text) - i), 0, -1):
                substring = text[i:i + j] # @inspect substring

                if substring in vocab:
                    result.append(substring) # @inspect result
                    i += j # @inspect i
                    matched = True
                    break

            if not matched:
                result.append("[UNK]")
                i += 1

        return result

    vocabulary = {
        "南京",
        "南京市",
        "南京市长",
        "长江",
        "长江大桥",
        "江",
        "大桥",
    }
    sentence = "🌁南京市长江大桥"
    tokens = max_match_tokenize(vocabulary, sentence) # @inspect tokens


def todo():
    text("[ ] 注册"), link(title="Github", url="https://github.com/"), text("账号")
    text("[ ] 申请学生认证，获得一系列开发工具"), link(title="GitHub Education", url="https://docs.github.com/en/education/about-github-education/github-education-for-students/apply-to-github-education-as-a-student")
    text("[ ] 使用GitHub Copilot"), link(title="GitHub Copilot", url="https://github.com/copilot")

def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
