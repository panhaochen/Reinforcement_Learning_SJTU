# 增强学习 · 作业 1：行为克隆与 DAgger

**2026 年秋季（Fall 2026）· 课程 95065**

课程平台与提交入口：[https://oc.sjtu.edu.cn/courses/95065](https://oc.sjtu.edu.cn/courses/95065)

## Setup

You can run this code on your own machine, on Kaggle or on Colab.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab option:** Open [hw1_colab.ipynb](hw1_colab.ipynb) in Google Colab. It will guide you through the entire workflow: environment setup, editing TODO code, training, and visualization. See [colab_guide.md](colab_guide.md) for a step-by-step Colab usage guide.
3. **Kaggle:** The first few sections of the notebook will install all required dependencies.

## Complete the code

Fill in sections marked with `TODO`. In particular, edit
 - [policies/MLP_policy.py](sjtuRL2026/policies/MLP_policy.py)
 - [infrastructure/utils.py](sjtuRL2026/infrastructure/utils.py)
 - [scripts/run_hw1.py](sjtuRL2026/scripts/run_hw1.py)

You have the option of running locally or on Colab using
 - [scripts/run_hw1.py](sjtuRL2026/scripts/run_hw1.py) (if running locally) or [hw1_colab.ipynb](hw1_colab.ipynb) (if running on Colab)
 - [scripts/run_hw1.ipynb](sjtuRL2026/scripts/run_hw1.ipynb) is for visualizing results after training

See the full assignment writeup: [hw1_writeup.md](hw1_writeup.md)（作业说明 PDF 的 Markdown 转写）. 更多细节以课程站点为准。

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python sjtuRL2026/scripts/run_hw1.py --expert_policy_file sjtuRL2026/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data sjtuRL2026/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```
(you can add --no_gpu at the end of the command)

Make sure to also try another environment.
See the [assignment writeup](hw1_writeup.md) for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python sjtuRL2026/scripts/run_hw1.py --expert_policy_file sjtuRL2026/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data sjtuRL2026/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```
(you can add --no_gpu at the end of the command)

Make sure to also try another environment.
See the [assignment writeup](hw1_writeup.md) for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](sjtuRL2026/scripts/run_hw1.ipynb) for more details.


## 提交作业

在 `hw1_template.tex` 中填写姓名、学号、答案、实验设置、表格、曲线和分析。使用 XeLaTeX 编译（建议运行两次 `xelatex hw1_template.tex`），最终只上传生成的 `hw1_template.pdf`，无需上传代码、日志、视频、TeX 源文件或 ZIP。Overleaf 用户请选择 XeLaTeX 编译器。

提交入口：[https://oc.sjtu.edu.cn/courses/95065](https://oc.sjtu.edu.cn/courses/95065)。
