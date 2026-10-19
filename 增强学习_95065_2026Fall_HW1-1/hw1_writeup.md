# 增强学习 · 作业 1：行为克隆与 DAgger

**RL2026fall · 增强学习 · Fall 2026**  
提交截止日期见课程站点。作业提交入口：[https://oc.sjtu.edu.cn/courses/95065](https://oc.sjtu.edu.cn/courses/95065)

---

## 1. 背景与目标

本作业实现**模仿学习**中的两类方法：

- **Behavior Cloning (BC)**：用专家数据 \(\mathcal{D} = \{(s_i, a_i)\}\) 做监督学习，最小化
  \[
  \min_\theta \;\mathbb{E}_{(s,a)\sim\mathcal{D}}\left[-\log \pi_\theta(a\mid s)\right]
  \]
  或等价的 MSE 损失（若策略输出为高斯均值等）。

- **DAgger (Dataset Aggregation)**：迭代地 (1) 用当前策略 \(\pi_\theta\) 在环境中采样轨迹；(2) 用专家策略对采样的状态重新标注动作；(3) 将新数据加入数据集并更新 \(\pi_\theta\)，以减轻分布偏移（distribution shift）。

---

## 2. 代码需要完成的部分

在以下文件中完成标记为 `TODO` 的部分：

- **`sjtuRL2026/policies/MLP_policy.py`**：MLP 策略的前向与监督学习更新（如 MSE 或交叉熵）。
- **`sjtuRL2026/infrastructure/utils.py`**：`sample_trajectories` / `sample_trajectory`，用于在环境中用给定策略采样轨迹。
- **`sjtuRL2026/scripts/run_hw1.py`**：
  - 第 1 次迭代：从 `expert_data` 加载专家轨迹进行 BC。
  - 第 2 次及以后（DAgger）：用 `utils.sample_trajectories` 采集 `params['batch_size']` 的转移；用专家策略对每条轨迹的 observation 重新标注 action，得到新的 `paths`。
  - 从 replay buffer 中采样 `train_batch_size` 的 \((s, a)\) 用于每步梯度更新。

---

## 3. Section 1：Behavior Cloning

- **目标**：仅用专家数据训练策略（\(n\_iter=1\)），不与环境交互。
- **命令示例**（Ant）：

```bash
python sjtuRL2026/scripts/run_hw1.py --expert_policy_file sjtuRL2026/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data sjtuRL2026/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```
(you can add --no_gpu at the end of the command)

- **基本要求**：至少在一个环境中运行 BC；调试时可保留 `--video_log_freq -1` 以关闭视频记录。需要提交结果时，可去掉该 flag 以生成视频。

- **结果汇报（表格）**：
  - 选择至少一个任务（如 `Ant-v4`），**重复多次 rollout 评估**你训练好的 BC 策略。
  - 在报告中用**表格**给出：该任务上策略回报的**平均值（mean）和标准差（std）**。表格中需标明：
    - 使用的环境名称（如 Ant-v4）；
    - 评估时的 `ep_len` 与 `eval_batch_size`；
    - 网络结构（层数、隐藏单元数）、使用的专家数据量、训练迭代次数等关键超参数。
  - 注意：为了“mean/std 有意义”，应确保 `eval_batch_size > ep_len`，这样每次评估会得到**多条轨迹**。例如：
    - 若 `ep_len = 1000` 且 `eval_batch_size = 5000`，则一次评估大约会得到 5 条轨迹（如有提前终止则更多），此时 log 中的 `Eval_AverageReturn`、`Eval_StdReturn` 就是这几条轨迹回报的均值和标准差。
    - 请在表格标题或说明中写明这些参数。

- **超参数实验（曲线）**：
  - 至少选取一个会显著影响 BC 性能的超参数进行实验，例如：
    - 训练步数 / 迭代次数（如 `num_agent_train_steps_per_iter` 或总梯度步数）；
    - 专家数据量（如使用专家数据子集 vs 全量）；
    - 网络结构（层数、宽度）等你认为合理的因素。
  - 固定其它设置不变，仅改变该超参数的取值，在**同一任务**（建议与上面表格同一个环境）上记录 BC 策略的性能（例如 `Eval_AverageReturn`）。
  - 画出「**BC 性能随该超参数变化的曲线**」，并在图注中说明：
    - 选取的超参数是什么、取值范围为何；
    - 选择该超参数的理由（为什么它会影响 BC 性能）；
    - 其他关键设置，如使用的环境、`ep_len`、`eval_batch_size`、训练轮次等。

- **交付**：将结果表格、性能曲线、实验设置和分析写入报告，最终只提交 PDF。

---

## 4. Section 2：DAgger

- **目标**：从 BC  warmstart 开始，多轮迭代：用当前策略采样 → 专家重新标注 → 加入 buffer 并训练。
- **命令示例**（Ant，10 轮）：

```bash
python sjtuRL2026/scripts/run_hw1.py --expert_policy_file sjtuRL2026/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data sjtuRL2026/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1 
```
(you can add --no_gpu at the end of the command)

- **基本要求**：至少在一个环境中运行 DAgger；建议选择你在 BC 部分已经使用过的 1–2 个任务，以便直接比较 BC 与 DAgger 的效果。

- **学习曲线实验**：
  - 对这 1–2 个任务运行 DAgger，多次迭代（例如 10 轮或更多），在每一轮结束后评估当前策略的性能。
  - 在报告中画出「**DAgger 迭代轮数 vs 策略平均回报**」的学习曲线：
    - 横轴：DAgger 迭代次数（如 0 表示纯 BC，1、2、… 表示加入若干轮 DAgger 后的策略）；
    - 纵轴：每轮评估得到的 `Eval_AverageReturn`；
    - 误差条：使用 `Eval_StdReturn` 作为标准差，反映多条评估轨迹上的波动。
  - 在同一张图上，用**水平线**画出：
    - 专家策略的表现（Expert policy）——使用给定的专家 policy 直接在环境中 roll out 得到的平均回报；
    - 纯 BC 策略的表现（Behavior Cloning）——DAgger 迭代前（第 0 轮）的策略回报。
  - 图注中需要说明：
    - 使用的任务/环境名称（如 Ant-v4、Walker2d-v4 等）；
    - 网络结构（层数、宽度）、专家数据量、训练超参数等关键信息；
    - 每轮评估时的 `ep_len` 与 `eval_batch_size`，以便读者知道 mean/std 是在多少条轨迹上统计出来的。

- **交付**：将 DAgger 学习曲线、实验设置和分析写入报告，最终只提交 PDF；实验日志自行保留。

---

## 5. 可视化* (可选)

使用 TensorBoard 查看 log：

```bash
tensorboard --logdir data
```

可查看标量曲线及（若开启）策略视频（Images 标签页）。指定多个 run：

```bash
tensorboard --logdir data/run1,data/run2,...
```

---

## 6. 提交说明

- 在 `hw1_template.tex` 中填写姓名、学号、答案、实验设置、结果表格、曲线和分析。
- 在模板目录运行两次 `xelatex hw1_template.tex`，生成 `hw1_template.pdf`；Overleaf 用户请选择 XeLaTeX 编译器。
- **最终只上传填写完整的 PDF 报告，无需上传代码、日志、视频、TeX 源文件或 ZIP。** 代码和日志自行保留。
- **提交入口**：[https://oc.sjtu.edu.cn/courses/95065](https://oc.sjtu.edu.cn/courses/95065)
- 具体格式、截止时间以课程/助教通知为准。

---

*本文档为 HW1 作业说明的 Markdown 版本，供 2026 年秋季《增强学习》课程使用。*
