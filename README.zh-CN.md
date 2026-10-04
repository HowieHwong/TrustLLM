<p align="center"><img src="images/logo.png" alt="TrustLLM" width="760"></p>
<p align="center"><a href="README.md">English</a> · <a href="https://arxiv.org/abs/2401.05561">论文</a> · <a href="https://howiehwong.github.io/TrustLLM/">文档</a> · <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">数据集</a></p>

**用同一套流程测试本地模型和 API 模型。** TrustLLM 围绕真实性、安全性、公平性、鲁棒性、隐私和机器伦理六个维度，提供数据下载、回复生成和评测工具。

## 三步开始

**安装**（不需要先克隆仓库）：

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

这是 GitHub 上的 0.4 源码版本；本次修改没有更新 PyPI 上旧的 `pip install trustllm` 发布包。正式实验建议把 `main` 替换成固定 commit SHA。

**下载数据**：

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**测试 API 模型**：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

将示例地址和模型名替换成实际服务信息。使用 OpenAI-compatible Chat Completions 接口；本地服务可填写 `http://localhost:8000/v1`。服务不要求密钥时可不设置密钥。

`--limit 5` 对每个数据文件取 5 条，用于试跑。打开 `runs/api-safety/report.html` 查看完成情况。报告显示生成进度，**不是可信度得分**。全量实验请去掉 `--limit`，并使用新的输出目录。

## 运行本地模型

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

模型权重首次使用时自动从 Hugging Face 下载，也可以把模型名替换成 `/path/to/checkpoint`。支持 `cpu`、`cuda:0`、`mps` 和 Accelerate 自动分配；支持的模型类型为 Transformers causal LM。模型访问权限、内存和架构仍需满足要求。

## 在 Python 中调用

```python
from trustllm import download_dataset, generate

download_dataset("data")
run = generate(
    model="your-model-id",
    backend="api",  # 本地模型改成 local，并填写 HF 模型 ID 或权重路径
    task="safety",
    data_path="data/dataset",
    output_dir="runs/python-safety",
    limit=5,
)
print(run["status"])
```

## 断点续跑和正式评分

原命令添加 `--resume` 即可重试失败样本、跳过成功样本；数据、生成参数和依赖版本必须一致。每次运行保留逐样本记录、输入哈希、配置和依赖版本。

对全量生成结果运行原有评分方法：

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

把路径改成自己的全量输出目录。评分结果写入 `scores.json`，并生成可直接打开的 `scores.html` 指标表。部分任务需要下载分类模型、调用 embedding 或 API judge；可用 `OPENAI_JUDGE_MODEL` 指定评审模型，API 可能产生费用。

[详细使用方法](docs/guides/running.md) · [评分说明](docs/guides/evaluation.md) · [评测维度与数据](docs/benchmark.md) · [贡献指南](CONTRIBUTING.md)

## 引用

研究中使用本项目时，请引用 [TrustLLM（ICML 2024）](https://openreview.net/forum?id=bWUU0LwwMp)，完整信息见 [CITATION.bib](CITATION.bib)。代码采用 [MIT](LICENSE) 许可证。
