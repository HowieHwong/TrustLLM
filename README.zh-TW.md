<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">論文</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">文件</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">資料集</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">排行榜</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <a href="README.md">English</a> &nbsp; / &nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp; / &nbsp;
  <strong>繁體中文</strong> &nbsp; / &nbsp;
  <a href="README.ja.md">日本語</a> &nbsp; / &nbsp;
  <a href="README.ko.md">한국어</a> &nbsp; / &nbsp;
  <a href="README.es.md">Español</a> &nbsp; / &nbsp;
  <a href="README.fr.md">Français</a>
</p>

TrustLLM 是一套開源研究工具，從**六個面向**評估大型語言模型的可信度。你可以使用本地權重或模型 API 執行 ICML 2024 基準測試，並統一保存資料、設定與結果。

想讓 AI Agent 自動執行評測？請參閱 [Agent 整合指南](docs/guides/agents.md)，了解 CLI/Python 呼叫及 JSON 結果，或從[機器可讀文件索引](https://howiehwong.github.io/TrustLLM/llms.txt)開始。

## 從這裡開始

**1 — 安裝。** 基礎套件支援透過 API 產生回覆及下載資料，不需要安裝 GPU 相關函式庫。

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

以上指令**直接從 GitHub 安裝 0.4 原始碼版本**，需要先安裝 Git。為確保實驗可重現，請將 `main` 替換為固定的 commit SHA。

**2 — 下載基準資料。**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — 測試 API 模型。** 設定服務所需的憑證與 API 位址：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

請將範例位址替換為服務供應商實際的 API 根位址。本地 OpenAI-compatible 服務可使用 `http://localhost:8000/v1` 及實際部署的模型 ID；若服務不要求驗證，可以不設定 API 金鑰。API 模式使用純文字 Chat Completions 介面。

開啟 `runs/api-safety/report.html` 查看完成情況。`--limit 5` 會從**每個資料檔案**取前五筆樣本進行初步測試；完整執行時請移除此參數，並使用新的輸出目錄。這份報告反映回覆產生的完成情況，並非基準評分。

## 本地權重，相同流程

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

模型權重會在首次使用時從 Hugging Face 下載。若已有權重，可將模型 ID 替換為 `/path/to/checkpoint`。`--device cpu`、`cuda:0` 與 `mps` 用於選擇裝置；`auto` 透過 Accelerate 分配模型。本地載入支援 Transformers 的因果語言模型，並在可用時採用 tokenizer 內建的聊天範本。模型存取權限、硬體容量與架構相容性仍須符合要求。

## 在 Python 中呼叫

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

API 設定來自環境變數 `OPENAI_API_KEY` 與 `OPENAI_BASE_URL`，也可以透過 Python 參數 `api_key` 與 `base_url` 明確傳入。JSON 設定、重試、模型版本、token 設定與中斷續跑的說明，請參閱[使用與設定指南](docs/guides/running.md)。

## 從回覆到評分

安裝評分所需的相依套件，再評估包含完整回覆的目錄：

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

將路徑替換為你的**完整執行**輸出目錄。結果儲存為 `scores.json` 與 `scores.html`；不完整的回覆會被拒絕。六個面向的評分流程保留原始基準方法。依任務而定，評分可能使用規則、下載的分類模型、嵌入向量或 API 評審模型。可透過 `OPENAI_JUDGE_MODEL` 指定帳號可用的評審模型；評審呼叫可能產生費用。完整執行前請閱讀[評分指南](docs/guides/evaluation.md)。

| 面向 | 評估內容 |
| :--- | :--- |
| **真實性** | 錯誤資訊 · 幻覺 · 迎合行為 |
| **安全性** | 越獄攻擊 · 濫用 · 過度安全行為 |
| **公平性** | 刻板印象 · 偏好 · 貶損 |
| **穩健性** | 對抗擾動 · 分布外輸入 |
| **隱私** | 隱私意識 · 資訊洩漏 |
| **倫理** | 道德判斷 · 道德選擇 |

[資料集與指標參考 →](docs/benchmark.md)

<details>
<summary><b>0.4 版本有哪些變更？</b></summary>

- API 與本地模型共用回覆產生流程，可直接指定模型 ID，不再依賴舊版模型白名單。
- 輕量的 API 安裝；本地推論、評分與封存相依套件分別透過 `local`、`eval` 與 `legacy` 安裝。
- 提供 `download`、`tasks`、`generate` 與 `evaluate` 指令，也可透過 `python -m trustllm` 呼叫。
- API 重試次數有上限；逐筆儲存檢查點、明確記錄失敗，並驗證 `--resume` 的相容條件。
- 回覆 JSON 仍與使用 `res` 欄位的既有評估器相容。
- 保存資料集雜湊、設定、相依套件版本、服務回傳的用量資訊，以及 HTML 回覆產生報告。

原始回覆產生引擎已封存於 `trustllm.generation.legacy`。輸入格式及模型整合方式已有變更，新執行結果不會自動等同於原論文的實驗設定。請參閱[遷移說明](docs/guides/running.md#migration-from-03)。

</details>

## 研究與開發

[CI 檢查](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [貢獻指南](CONTRIBUTING.md) · [設計參考](docs/design.md) · [更新紀錄](docs/changelog.md) · [問題回報](https://github.com/HowieHwong/TrustLLM/issues)

本頁為 README 的翻譯；詳細指南目前以英文提供。文件翻譯不會變更基準提示詞、資料集或評分方法。

若 TrustLLM 對你的研究有所幫助，請引用 [ICML 2024 論文](https://openreview.net/forum?id=bWUU0LwwMp)。完整 BibTeX 請見 [CITATION.bib](CITATION.bib)。程式碼採用 [MIT](LICENSE) 授權；資料集仍適用各原始來源的條款。
