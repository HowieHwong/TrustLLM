<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">論文</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">ドキュメント</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">データセット</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">リーダーボード</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <a href="README.md">English</a> &nbsp; / &nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp; / &nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp; / &nbsp;
  <strong>日本語</strong> &nbsp; / &nbsp;
  <a href="README.ko.md">한국어</a> &nbsp; / &nbsp;
  <a href="README.es.md">Español</a> &nbsp; / &nbsp;
  <a href="README.fr.md">Français</a>
</p>

TrustLLM は、大規模言語モデルの信頼性を**6つの観点**から評価するオープンソースの研究用ツールキットです。ローカルのモデル重みでもモデル API でも ICML 2024 のベンチマークを実行でき、データ・設定・結果をまとめて管理できます。

## はじめに

**1 — インストール。** 基本パッケージには API 経由の応答生成とデータのダウンロード機能が含まれます。GPU 関連のライブラリは不要です。

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

このコマンドでインストールされるのは **0.4 のソース版**です。今回の更新は PyPI には公開されていないため、`pip install trustllm` では旧版がインストールされる場合があります。実験を再現できるようにするには、`main` を固定のコミット SHA に置き換えてください。

**2 — ベンチマークデータを取得。**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — API モデルを試す。** 利用するサービスの認証情報とエンドポイントを設定します。

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

例の URL は、プロバイダーの実際の API ベース URL に置き換えてください。ローカルの OpenAI 互換サーバーでは、`http://localhost:8000/v1` とサーバーで提供しているモデル ID を指定できます。サーバーが認証を要求しない場合、API キーは不要です。API モードはテキストのみの Chat Completions を使用します。

`runs/api-safety/report.html` を開くと処理の完了状況を確認できます。`--limit 5` は動作確認のため、**各データファイルから**先頭の5件を選びます。全件を実行する場合はこの指定を外し、新しい出力ディレクトリを使ってください。このレポートは応答生成の進捗を示すもので、ベンチマークのスコアではありません。

## ローカルの重みでも同じ手順

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

Hugging Face のモデル重みは初回利用時にダウンロードされます。保存済みの重みを使う場合は、モデル ID を `/path/to/checkpoint` に置き換えてください。`--device cpu`、`cuda:0`、`mps` でデバイスを指定でき、`auto` では Accelerate がモデルを配置します。ローカルでは Transformers の因果言語モデルに対応し、利用可能な場合はトークナイザーのチャットテンプレートを使います。モデルへのアクセス権、必要なメモリ容量、アーキテクチャの互換性は別途満たす必要があります。

## Python から使う

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

API の設定には `OPENAI_API_KEY` と `OPENAI_BASE_URL` を使います。Python の引数 `api_key` と `base_url` で直接指定することもできます。JSON 設定、再試行、モデルのリビジョン、トークン設定、中断した実行の再開については、[実行・設定ガイド](docs/guides/running.md)を参照してください。

## 応答を採点する

採点用の依存パッケージをインストールし、生成済みの応答がそろったディレクトリを評価します。

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

パスは**全件実行**の出力ディレクトリに置き換えてください。結果は `scores.json` と `scores.html` に保存され、不完全な応答は受け付けません。6つの評価パイプラインは元のベンチマーク手法を維持しています。タスクによって、ルール、ダウンロードした分類モデル、埋め込み、API 経由の評価モデルを使用します。利用可能な評価モデルは `OPENAI_JUDGE_MODEL` で指定できます。評価モデルの呼び出しには料金が発生する場合があります。全件実行の前に[採点ガイド](docs/guides/evaluation.md)を確認してください。

| 観点 | 評価対象 |
| :--- | :--- |
| **真実性** | 誤情報・ハルシネーション・迎合 |
| **安全性** | ジェイルブレイク・悪用・過剰な安全行動 |
| **公平性** | ステレオタイプ・選好・蔑視 |
| **頑健性** | 敵対的摂動・分布外の入力 |
| **プライバシー** | プライバシーへの認識・情報漏洩 |
| **倫理** | 道徳的判断・道徳的選択 |

[データセットと評価指標 →](docs/benchmark.md)

<details>
<summary><b>0.4 の主な変更点</b></summary>

- API とローカル生成に共通の実行基盤を導入。モデル ID をそのまま指定でき、旧来の許可リストは不要です。
- API 用の軽量インストールに加え、`local`、`eval`、アーカイブ用の `legacy` を選択できます。
- `download`、`tasks`、`generate`、`evaluate` コマンドを提供。`python -m trustllm` からも利用できます。
- API 再試行回数に上限を設定。サンプルごとのチェックポイント、明示的な失敗記録、互換性を確認する `--resume` を追加しました。
- 応答 JSON は、既存の `res` フィールドを使う評価器と互換性があります。
- データセットのハッシュ、設定、依存パッケージのバージョン、返された使用量情報、HTML の生成レポートを保存します。

元の生成エンジンは `trustllm.generation.legacy` にアーカイブされています。入力の書式とモデル連携方法が変わっているため、新しい実行が元論文の実験設定と一致するとは限りません。[移行ガイド](docs/guides/running.md#migration-from-03)を参照してください。

</details>

## 研究と開発

[CI](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [貢献ガイド](CONTRIBUTING.md) · [設計の参考資料](docs/design.md) · [変更履歴](docs/changelog.md) · [Issue](https://github.com/HowieHwong/TrustLLM/issues)

このページは README の翻訳です。詳細ガイドは現在英語で提供しています。ドキュメントの翻訳によって、ベンチマークのプロンプト、データセット、採点方法が変わることはありません。

研究で TrustLLM を利用した場合は、[ICML 2024 の論文](https://openreview.net/forum?id=bWUU0LwwMp)を引用してください。完全な BibTeX は [CITATION.bib](CITATION.bib) にあります。コードは [MIT ライセンス](LICENSE) で公開しています。データセットには各提供元の利用条件が適用されます。
