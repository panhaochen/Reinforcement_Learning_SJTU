# 增强学习 · 作业 2：策略梯度

**2026 年秋季（Fall 2026）· 课程 95065**

课程平台与提交入口：[https://oc.sjtu.edu.cn/courses/95065](https://oc.sjtu.edu.cn/courses/95065)

## 最终提交要求

请完成代码与实验，把要求的图表、命令配置和文字回答填写到 `hw2_template.tex`，并使用任意能够正确编译本模板的 LaTeX 环境生成 `hw2_template.pdf`。不限定具体编译器。

**课程平台最终只需上传填写完整、编译成功且能够正常打开的 PDF 报告。无需上传代码、实验数据、日志、模型、TeX 源文件或 ZIP 压缩包。** 代码和实验输出请自行保留，仅用于完成实验和生成报告。

截止日期以课程平台通知为准。

## Setup

See [installation.md](installation.md). It's worth going through this again since some dependencies have changed since homework 1. You also need to make sure to run `pip install -e .` in the hw2 folder.


## Complete the code

There are TODOs in these files:

- `sjtuRL2026/scripts/run_hw2.py`
- `sjtuRL2026/agents/pg_agent.py`
- `sjtuRL2026/networks/policies.py`
- `sjtuRL2026/networks/critics.py`
- `sjtuRL2026/infrastructure/utils.py`
- `sjtuRL2026\scripts\run_hw2.ipynb`

### note: The last file is used to generate images for analysis.

### note 2: If need TensorBoard, uncomment the corresponding `tensorboard` line in `requirements.txt`.

## note 3: box2d-py 可能有一些问题，请忽略。

See [hw2_template.tex](hw2_template.tex) for the complete assignment and report template.
