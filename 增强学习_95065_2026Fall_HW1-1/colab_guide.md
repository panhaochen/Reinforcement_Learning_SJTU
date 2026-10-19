# Google Colab 使用指南

本指南帮助你在 Google Colab 上完成 HW1 作业。

## 前置准备

- 一个 Google 账号
- 推荐使用 Chrome 或 Edge 浏览器
- 稳定的网络连接

## 步骤 1：将代码上传到 Google Drive

1. 将整个 `hw1` 文件夹压缩为 `hw1.zip`
2. 打开 [Google Drive](https://drive.google.com)，上传 `hw1.zip` 或整个 `hw1` 文件夹
3. 记住上传的路径（如 `MyDrive/hw1`）

## 步骤 2：打开 Colab Notebook

**方式 A：直接上传 notebook**
1. 打开 [Google Colab](https://colab.research.google.com)
2. 点击 `File -> Upload notebook`
3. 上传 `hw1_colab.ipynb`

**方式 B：从 Google Drive 打开**
1. 将 `hw1_colab.ipynb` 上传到 Google Drive
2. 在 Drive 中双击该文件，选择用 Google Colaboratory 打开

## 步骤 3：切换 GPU Runtime

1. 点击菜单栏 `Runtime -> Change runtime type`
2. 在 `Hardware accelerator` 下拉菜单中选择 `T4 GPU`
3. 点击 `Save`

> GPU 不是必须的（CPU 也能运行），但 GPU 可以显著加速训练过程。

## 步骤 4：运行环境配置

按顺序运行 **Part 0** 中的所有 cell：
1. 安装系统依赖（xvfb）
2. 安装 Python 包（mujoco, gym 等）
3. 上传代码（选择 Drive 挂载或 zip 上传）
4. 安装项目包（`pip install -e .`）
5. 配置 MuJoCo 渲染
6. 验证安装

## 步骤 5：编辑 TODO 代码

在 Colab 中编辑 `.py` 文件：

1. 点击左侧边栏的 **文件夹图标** 打开文件浏览器
2. 导航到需要编辑的文件（如 `sjtuRL2026/infrastructure/utils.py`）
3. **双击文件** 在编辑器中打开
4. 找到 `TODO` 标记，填写你的实现代码
5. 按 `Ctrl+S` 保存文件

需要编辑的 3 个文件：
- `sjtuRL2026/infrastructure/utils.py`
- `sjtuRL2026/policies/MLP_policy.py`
- `sjtuRL2026/scripts/run_hw1.py`

## 步骤 6：运行训练 & 可视化

按顺序运行 **Part 2**（训练）和 **Part 3**（可视化）中的 cell。

## 步骤 7：下载结果

运行 **Part 4** 中的 cell，将 `data/` 目录打包下载，作为个人备份。此压缩包无需提交。将实验结果填写到 `hw1_template.tex`，使用 XeLaTeX 编译后，只提交生成的 PDF 报告。

> Colab 的运行时存储是临时的！每次会话结束后文件会丢失，请务必下载你的训练结果。

## 常见问题

### Q: MuJoCo 渲染报错 `GLEW initialization error`
**A:** 确保已运行以下代码：
```python
import os
os.environ['MUJOCO_GL'] = 'egl'
```
这需要在 `import gym` 之前设置。如果仍有问题，重启 runtime 后重新运行。

### Q: 内存不足 (OOM)
**A:** Colab 免费版提供约 12GB RAM。如果内存不足：
- 减小 `--batch_size`（如改为 500）
- 减少 `--n_iter`
- 在 `Runtime -> Manage sessions` 中关闭其他不使用的 notebook

### Q: GPU 配额用完
**A:** Colab 免费版有每日 GPU 使用限额。你可以：
- 等待配额恢复（通常几小时后）
- 使用 CPU 运行（添加 `--no_gpu` 参数）
- 使用 Colab Pro 获取更多配额

### Q: 安装包时报版本冲突
**A:** Colab 预装了很多包。我们的安装命令已避免覆盖 Colab 自带的 PyTorch。如果仍有冲突，尝试重启 runtime（`Runtime -> Restart runtime`）后重新安装。

### Q: `pip install -e .` 报错找不到 `setup.py`
**A:** 确保你已经 `cd` 到了正确的 `hw1` 目录。检查当前工作目录：
```python
import os
print(os.getcwd())
```

### Q: 训练中断或 Colab 断开连接
**A:** Colab 空闲一段时间后会自动断开。建议：
- 保持浏览器标签页活跃
- 定期检查训练进度
- 使用较少的迭代次数先验证代码正确性
