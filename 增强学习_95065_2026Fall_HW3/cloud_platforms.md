# 运行平台选择

本课程没有分配 Google Cloud project，也没有课程专属的云计算额度。无需创建 Google Cloud 结算账户来完成本作业。

可以在本地 Linux、Colab 或自行租用的 AutoDL 等国内云服务器运行。平台选择不影响评分。

## Colab

访问 https://colab.research.google.com/ ，创建笔记本。上传作业压缩包，解压后进入作业目录，按 installation.md 安装依赖。需要 GPU 时可在运行时设置中选择 GPU。

注意：本作业采用 Python 3.10 与旧版 Gym/PyTorch 依赖。Colab 默认 Python 版本可能不兼容，不能直接假定当前预装环境可用；请使用独立 Python 3.10 环境并从该环境运行命令，若无法配置，建议改用本地或 AutoDL。不要仅修改版本号后混用 Gymnasium API。Colab 资源及 GPU 类型并不固定，运行环境可能中断；及时下载报告图表和训练输出。

## AutoDL 等国内云服务器

自行注册并按平台计费，不存在课程统一项目或赠送额度。优先选 Linux、Python 3.10 的环境，GPU 按任务需求选择，并用 nvidia-smi 检查设备。将作业解压至 /root/autodl-tmp/hw3，再按 installation.md 创建独立环境。先完成 CartPole 或 Pendulum 小规模检查，再运行长实验。留意磁盘容量，训练后备份重要输出并停止不使用的实例。

平台参考：https://www.autodl.com/docs/ ；Colab资源说明：https://research.google.com/colaboratory/faq.html
