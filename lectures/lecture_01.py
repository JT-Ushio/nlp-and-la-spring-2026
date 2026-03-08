from execute_util import link, image, text

def main():
    text("# FORE20066：自然语言处理与语言习得 \n## Natural Language Processing and Language Acquisition")
    image("images/stuff.png", width=600)

    text("| 单位 | 专业/部门 | 经历 | 研究方向 |\n| :---: | :---: | :---: | :---: |\n| 华东师范大学 | 计算机科学与技术 | 学士（保送） | 《基于循环神经网络的依存句法分析》 |\n| 华东师范大学 | 计算机科学与技术 | 博士（硕博连读） | 《多语言依存句法分析》 |\n| 复旦大学 | 自然语言处理实验室 | 博士后研究人员 |  |\n| **复旦大学** | **外文学院** | **助理教授** | **计算语言学与大语言模型** |\n| **上海人工智能实验室** | **大模型中心** | **算法顾问（兼）** | |\n")

    text("- ACL/NeurIPS/ICLR/EMNLP等自然语言处理和深度学习顶会发表论文30余篇")
    text("- TA@《统计自然语言处理》，TA@《深度学习导论》，代课教师@《自然语言处理导论》")
    text("- 网页编译框架来源Stanford CS336课程")

    why_language_and_cs_matter()
    intro()
    todo()

    Q_and_A()

def why_language_and_cs_matter():
    image("images/block_news.png", width=600)
    text("- **利润增长 → 裁员近半 → 股价飙升** "), link(title="[WallStreet]", url="https://wallstreetcn.com/articles/3766301")
    text("- **AI时代下，企业通过裁员来突出效率与价值**")
    text("---")
    image("images/hc_devlop.png", width=600)
    text("- **软件开发行业**的青年就业岗位持续萎缩 "), link(title="[Stanford Report, Fig 1]", url="https://digitaleconomy.stanford.edu/app/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf")
    text("---")
    image("images/hc_custom_service.png", width=600)
    text("- **社会服务行业**的青年就业岗位持续萎缩 "), link(title="[Stanford Report, Fig 1]", url="https://digitaleconomy.stanford.edu/app/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf")
    text("---")
    image("images/hc_other_job.png", width=600)
    text("- **销售经理、产线员工、护理师**等AI不可替代（目前）行业的就业岗位正常增长 "), link(title="[Stanford Report, Fig A2]", url="https://digitaleconomy.stanford.edu/app/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf")
    text("---")
    image("images/swe_bench.png", width=600)
    text("- **基础软件开发能力评测集**性能突飞猛进：4.4%(2023) → 71.7%(2024) "), link(title="[AI Index Report 2025, Fig 2.5.4]", url="https://arxiv.org/pdf/2504.07139")
    text("---")
    image("images/math_hard_case.png", width=600)
    image("images/math_hard_bench.png", width=600)
    text("- **前沿数学问题解题**性能显著提升 "), link(title="[AI Index Report 2025, Fig 2.6.7]", url="https://arxiv.org/pdf/2504.07139")
    text("---")
    image("images/math.png", width=600)
    text("- 大型语言模型(LLM)在国际数学奥林匹克(IMO)2025中取得了相当于金牌的成绩。")
    image("images/acmicpc1.png", width=400)
    text("- 大型语言模型(LLM)在2025年ACM国际大学生程序设计竞赛(ACM-ICPC)中表现出色，达到了世界级水平。")
    image("images/acmicpc2.png", width=900)
    text("- 12道题里11道题OpenAI都是一次提交就通过，只有最难的G题，花了9次才提交通过。 ")
    image("images/acmicpc3.png", width=600)
    text("- A题：编程设计一个数据结构：斜堆（skew heap）......")
    text("---")
    text("### AI作为生产力工具，难替代、全栈、善用AI的求职者具有竞争力")
    text("#### 回顾：AI的首次出圈是AlphaGo")
    image("images/go1.jpg", width=600)
    text("- 2016年AlphaGo 4:1 战胜围棋世界冠军李世石")
    image("images/go2.jpg", width=600)
    text("- 2017年AlphaGo 3:0 战胜围棋世界冠军柯洁")
    image("images/go_and_ai.png", width=600)
    text("=> 棋手利用AI快速提升水平 "), link(title="[LizzieYzy]", url="https://blog.csdn.net/gitblog_00913/article/details/158304469")

    text("### AI确实很重要，但为什么外语+计算机？🤔")
    text("#### 举例：中美模型最大的差距在于基于人类反馈的强化学习(RLHF)")
    image("images/6-sft-rlhf.png", width=600)
    text("- 大规模预训练 → 指令微调 → 基于人类反馈的强化学习")
    text("- “暴力求解”只适用于通用、数据量大、数据质量高的任务")
    text("- 语言学家能够帮助设计更有效的反馈机制，从而提升AI的专业性和适应性")
    text("- 例如：创造一个会说脱口秀的AI")

    text("- 危机：“暴力求解”但互联网数据已经用完")
    image("images/limits_of_llm_data.png", width=600)
    text("- 语言学家能够帮助合成有价值的数据，解决数据危机")
    text("### 自然语言处理（NLP）是一门交叉学科，深度融合了计算机科学、语言学、统计学等多个领域的知识和技术。")
    text("- **AI应用**的研究依赖交叉学科的思维与技能")
    text("- **AI进步**的研究依赖交叉学科的思维与技能")


def intro():

    text("### 2024年 AI主导的研究获得了诺贝尔物理学奖、化学奖")
    image("images/nbe1.png", width=600)
    text("- 诺贝尔物理学奖表彰“利用人工神经网络实现机器学习的基础性发现和发明”")
    image("images/nbe2.jpg", width=600)
    text("- 诺贝尔化学奖表彰“利用人工智能在复杂化学反应预测中的突破性贡献”")

    text("### NLP旨在探索实现人与计算机之间用自然语言进行有效交流的理论与方法。")
    text("-  自然语言理解：能够理解自然语言的意义")
    text("-  自然语言生成：以自然语言文本来表达给定的意图、思想等")

    text("### 自然语言处理的研究内容十分庞杂。")
    image("images/nlp_domain.png", width=600)
    text("-  整体：基础算法研究和应用技术研究")
    text("-  语言单位角度：字、词、短语、句子、段落以及篇章等不同粒度")
    text("-  语言学研究角度：形态学、语法学、语义学、语用学等不同层面")
    text("-  机器学习方法层面：有监督、无监督、半监督、强化学习等")


    text("### 自然语言处理难的根本原因：自然语言在各个层面都广泛存在的各种各样的歧义性。")
    text("-  语音歧义：chéng shì：城市、程式、成事、城事")
    text("-  词义歧义：bank：银行、河岸")
    text("-  分词歧义：南京市长江大桥")
    text("-  句法歧义：I saw the man with the telescope.（我用望远镜看见了那个男人；我看见了那个带着望远镜的男人）")
    text("-  语义歧义：The chicken is ready to eat.（鸡准备吃了；鸡准备被吃了）")
    text("-  语用歧义：Can you pass the salt?（你能递一下盐吗？；你能把盐递给我吗？）")


    text("### 1. 基于规则的方法（Rule-based Methods）")
    image("images/rule_based.png", width=600)

    text("### 2. 基于模型的方法（Model-based Methods）")
    image("images/model_based.png", width=600)

    text("### 3. 基于表示的方法（Representation-based Methods）")
    image("images/representation_based.png", width=600)

    text("### 4. 基于预训练模型的方法（Pretrained Model-based Methods）")
    image("images/plm_based.png", width=600)

    text("**任务共享的比例在不断增加**")

    text("### ACL：自然语言处理领域的顶级会议，涵盖了自然语言处理的各个方面，包括基础算法、应用技术、语言学研究等。")
    text("ACL 2021：没有一个直接关于LLM的track")
    image("images/acl_2021.png", width=600)
    text("ACL 2026：增加多个直接关于LLM的track")
    image("images/acl_2026.png", width=600)

    text("### 课程目标：大语言模型为主，NLP传统任务为辅")
    text("### 课程目标：语言不仅以文本的形式存在和学习，还以语音、图像等多模态的形式存在")

    text("### 课程大纲")
    image("images/course_outline.png", width=600)

    text("### 课程考核")
    text("- 出勤 & 课堂表现：20%")
    text("- 课程实践：50%（三次）")
    text("- 期末考试：30%")

    text("- CFFF国内高校最大的AI计算集群")
    image("images/cfff1.png", width=400)
    image("images/cfff2.png", width=400)

    text("### 参考教材")
    text("- 张奇、桂韬、郑锐、黄萱菁"), link(title="《大规模语言模型：从理论到实践》", url="https://intro-llm.github.io/")
    text("- 张奇、桂韬、黄萱菁"), link(title="《自然语言处理导论》", url="https://intro-nlp.github.io/")


def todo():
    text("[ ] 注册"), link(title="Github", url="https://github.com/"), text("账号")
    text("[ ] 申请学生认证，获得一系列开发工具"), link(title="GitHub Education", url="https://docs.github.com/en/education/about-github-education/github-education-for-students/apply-to-github-education-as-a-student")
    text("[ ] 使用GitHub Copilot"), link(title="GitHub Copilot", url="https://github.com/copilot")

def Q_and_A():
    text("# 问答环节：欢迎大家提问！")
    image("images/stuff.png", width=600)


if __name__ == "__main__":
    main()
