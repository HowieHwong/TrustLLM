<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">论文</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">文档</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">数据集</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">排行榜</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <a href="README.md">English</a> &nbsp; / &nbsp;
  <strong>简体中文</strong> &nbsp; / &nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp; / &nbsp;
  <a href="README.ja.md">日本語</a> &nbsp; / &nbsp;
  <a href="README.ko.md">한국어</a> &nbsp; / &nbsp;
  <a href="README.es.md">Español</a> &nbsp; / &nbsp;
  <a href="README.fr.md">Français</a>
</p>

TrustLLM 是一个开源研究工具包，从**六个维度**评估大语言模型的可信度。你可以使用本地权重或模型 API 运行 ICML 2024 基准，并统一保存数据、配置和结果。

想让 AI Agent 自动运行评测？查看 [Agent 集成指南](docs/guides/agents.md)，了解 CLI/Python 调用及 JSON 结果，或从[机器可读文档索引](https://howiehwong.github.io/TrustLLM/llms.txt)开始。

## 从这里开始

**1 — 安装。** 基础包支持 API 回复生成和数据下载，无需安装 GPU 相关库。

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

以上命令安装的是 **0.4 源码版本**。本次更新没有发布到 PyPI；直接执行 `pip install trustllm` 仍可能安装旧版。为保证实验可复现，请将 `main` 替换为固定的 commit SHA。

**2 — 下载基准数据。**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — 测试 API 模型。** 设置所用服务的凭据和接口地址：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

请将示例地址替换为服务商实际的 API 根地址。对于本地 OpenAI-compatible 服务，可使用 `http://localhost:8000/v1` 和实际部署的模型 ID；如果服务不要求身份验证，可以不设置 API 密钥。API 模式使用纯文本 Chat Completions 接口。

打开 `runs/api-safety/report.html` 查看完成情况。`--limit 5` 会从**每个数据文件**中取前五条样本进行试跑；正式全量运行时请去掉此参数，并使用新的输出目录。这份报告反映生成完成情况，不是基准评分。

## 本地权重，同一套流程

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

首次使用时，模型权重会从 Hugging Face 下载。已有权重可通过 `/path/to/checkpoint` 指定。`--device cpu`、`cuda:0` 和 `mps` 用于选择设备；`auto` 使用 Accelerate 分配模型。本地加载支持 Transformers 的因果语言模型，并在可用时采用 tokenizer 自带的聊天模板。模型访问权限、硬件容量和架构兼容性仍需满足要求。

## 在 Python 中调用

```python
from trustllm import download_dataset, generate

download_dataset("data")

run = generate(
    model="your-model-id",
    backend="api",                    # Switch to "local" for HF weights.
    task="safety",
    data_path="data/dataset",
    output_dir="runs/python-safety",
    limit=5,
)
print(run["status"], run["successful"], run["total"])
```

API 配置来自环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`，也可以通过 Python 参数 `api_key` 和 `base_url` 显式传入。JSON 配置、重试、模型版本、token 设置和断点续跑的说明见[使用与配置指南](docs/guides/running.md)。

## 从回复到评分

安装评分依赖，然后评估包含完整生成结果的目录：

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

将路径替换为你的**全量运行**输出目录。结果保存为 `scores.json` 和 `scores.html`；不完整的回复会被拒绝。六个维度的评分流程保留原基准方法。不同任务可能使用规则、下载的分类器、embedding 或 API 评审模型。可通过 `OPENAI_JUDGE_MODEL` 指定账号可用的评审模型；评审调用可能产生费用。全量运行前请阅读[评分指南](docs/guides/evaluation.md)。

| 维度 | 评估内容 |
| :--- | :--- |
| **真实性** | 错误信息 · 幻觉 · 迎合行为 |
| **安全性** | 越狱攻击 · 滥用 · 过度安全行为 |
| **公平性** | 刻板印象 · 偏好 · 贬损 |
| **鲁棒性** | 对抗扰动 · 分布外输入 |
| **隐私** | 隐私意识 · 信息泄露 |
| **伦理** | 道德判断 · 道德选择 |

[数据集与指标参考 →](docs/benchmark.md)

<details>
<summary><b>0.4 版本有哪些变化？</b></summary>

- API 和本地生成共用运行入口，直接接受模型 ID，不再依赖旧的模型白名单。
- 轻量 API 安装；本地、评分和归档依赖分别通过 `local`、`eval` 和 `legacy` 安装。
- 提供 `download`、`tasks`、`generate` 和 `evaluate` 命令，也可通过 `python -m trustllm` 调用。
- API 重试次数有上限；逐样本保存检查点，明确记录失败，并检查 `--resume` 的兼容条件。
- 输出 JSON 保持与基于 `res` 字段的现有评估器兼容。
- 保存数据集哈希、运行配置、依赖版本、服务返回的用量信息及 HTML 生成报告。

原生成引擎归档在 `trustllm.generation.legacy`。输入格式和模型接入方式发生了变化，因此新运行不自动等同于论文中的原始实验设置。请参阅[迁移说明](docs/guides/running.md#migration-from-03)。

</details>

## 研究与开发

[CI 检查](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [贡献指南](CONTRIBUTING.md) · [设计参考](docs/design.md) · [更新记录](docs/changelog.md) · [问题反馈](https://github.com/HowieHwong/TrustLLM/issues)

本页是 README 的翻译；详细指南目前以英文提供。文档翻译不会更改基准提示词、数据集或评分方法。

如果 TrustLLM 对你的研究有帮助，请引用 [ICML 2024 论文](https://openreview.net/forum?id=bWUU0LwwMp)。完整 BibTeX 见 [CITATION.bib](CITATION.bib)。代码采用 [MIT](LICENSE) 许可证；数据集仍适用各原始来源的条款。
