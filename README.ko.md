<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">논문</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">문서</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">데이터셋</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">리더보드</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <a href="README.md">English</a> &nbsp; / &nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp; / &nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp; / &nbsp;
  <a href="README.ja.md">日本語</a> &nbsp; / &nbsp;
  <strong>한국어</strong> &nbsp; / &nbsp;
  <a href="README.es.md">Español</a> &nbsp; / &nbsp;
  <a href="README.fr.md">Français</a>
</p>

TrustLLM은 대규모 언어 모델의 신뢰성을 **여섯 가지 차원**에서 평가하는 오픈소스 연구 도구입니다. 로컬 모델 가중치나 모델 API로 ICML 2024 벤치마크를 실행하고, 데이터·설정·결과를 함께 관리할 수 있습니다.

## 시작하기

**1 — 설치.** 기본 패키지는 API 응답 생성과 데이터 다운로드를 지원합니다. GPU 관련 라이브러리는 필요하지 않습니다.

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

이 명령은 **0.4 소스 버전**을 설치합니다. 이번 변경 사항은 PyPI에 배포되지 않았으므로 `pip install trustllm`으로는 이전 버전이 설치될 수 있습니다. 실험의 재현성을 확보하려면 `main`을 특정 커밋 SHA로 바꾸세요.

**2 — 벤치마크 데이터 다운로드.**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — API 모델 실행.** 사용할 서비스의 인증 정보와 엔드포인트를 설정하세요.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

예시 URL을 서비스 제공자의 실제 API 기본 주소로 바꾸세요. 로컬 OpenAI 호환 서버를 사용한다면 `http://localhost:8000/v1`과 해당 서버의 모델 ID를 지정할 수 있습니다. 서버가 인증을 요구하지 않으면 API 키는 생략할 수 있습니다. API 모드는 텍스트 전용 Chat Completions 인터페이스를 사용합니다.

`runs/api-safety/report.html`을 열면 완료 현황을 확인할 수 있습니다. `--limit 5`는 동작 확인을 위해 **각 데이터 파일에서** 처음 다섯 개의 샘플을 선택합니다. 전체 실행 시에는 이 옵션을 제거하고 새 출력 디렉터리를 사용하세요. 이 보고서는 응답 생성 현황을 보여 주며, 벤치마크 점수가 아닙니다.

## 로컬 가중치도 같은 방식으로

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

Hugging Face 모델 가중치는 처음 사용할 때 다운로드됩니다. 저장된 가중치를 사용하려면 모델 ID 대신 `/path/to/checkpoint`를 지정하세요. `--device cpu`, `cuda:0`, `mps`로 장치를 선택할 수 있으며, `auto`는 Accelerate를 통해 모델을 배치합니다. 로컬 로딩은 Transformers의 인과 언어 모델을 지원하고, 토크나이저에 채팅 템플릿이 있으면 이를 적용합니다. 모델 접근 권한, 하드웨어 용량, 아키텍처 호환성은 별도로 충족해야 합니다.

## Python에서 사용하기

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

API 설정은 `OPENAI_API_KEY`와 `OPENAI_BASE_URL` 환경 변수에서 가져옵니다. Python의 `api_key`, `base_url` 인자로 직접 지정할 수도 있습니다. JSON 설정, 재시도, 모델 리비전, 토큰 설정, 중단된 실행 재개에 관한 설명은 [사용 및 설정 가이드](docs/guides/running.md)를 참고하세요.

## 응답을 점수로 변환하기

채점용 의존성을 설치한 뒤, 생성된 응답이 모두 들어 있는 디렉터리를 평가하세요.

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

경로를 **전체 실행**의 출력 디렉터리로 바꾸세요. 결과는 `scores.json`과 `scores.html`에 저장되며, 불완전한 응답은 거부됩니다. 여섯 가지 평가 파이프라인은 기존 벤치마크 방법을 유지합니다. 작업에 따라 규칙, 다운로드한 분류 모델, 임베딩 또는 API 평가 모델을 사용합니다. 계정에서 이용할 수 있는 평가 모델을 `OPENAI_JUDGE_MODEL`로 지정하세요. 평가 모델 호출에는 비용이 발생할 수 있습니다. 전체 실행 전 [채점 가이드](docs/guides/evaluation.md)를 확인하세요.

| 평가 차원 | 평가 내용 |
| :--- | :--- |
| **진실성** | 잘못된 정보 · 환각 · 사용자 의견에 대한 무조건적 동조 |
| **안전성** | 탈옥 공격 · 악용 · 과도한 안전 행동 |
| **공정성** | 고정관념 · 선호 · 비하 |
| **강건성** | 적대적 교란 · 분포 밖 입력 |
| **프라이버시** | 프라이버시 인식 · 정보 유출 |
| **윤리** | 도덕적 판단 · 도덕적 선택 |

[데이터셋 및 평가 지표 →](docs/benchmark.md)

<details>
<summary><b>0.4의 주요 변경 사항</b></summary>

- API와 로컬 생성에 공통 실행기를 사용합니다. 모델 ID를 직접 지정할 수 있어 이전의 모델 허용 목록에 의존하지 않습니다.
- 가벼운 API 설치를 제공하고, `local`, `eval`, 보관된 엔진용 `legacy` 의존성을 선택적으로 설치할 수 있습니다.
- `download`, `tasks`, `generate`, `evaluate` 명령을 제공합니다. `python -m trustllm`으로도 실행할 수 있습니다.
- API 재시도 횟수를 제한하고, 샘플별 체크포인트와 명시적인 실패 기록, 호환성을 검사하는 `--resume`을 제공합니다.
- 응답 JSON은 기존 `res` 필드 기반 평가기와 호환됩니다.
- 데이터셋 해시, 설정, 의존성 버전, 서비스가 반환한 사용량 정보, HTML 생성 보고서를 저장합니다.

기존 생성 엔진은 `trustllm.generation.legacy`에 보관되어 있습니다. 입력 형식과 모델 연동 방식이 변경되었으므로 새 실행이 원 논문의 실험 설정과 자동으로 일치하지는 않습니다. [마이그레이션 안내](docs/guides/running.md#migration-from-03)를 참고하세요.

</details>

## 연구 및 개발

[CI 검사](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [기여 가이드](CONTRIBUTING.md) · [설계 참고 자료](docs/design.md) · [변경 이력](docs/changelog.md) · [이슈](https://github.com/HowieHwong/TrustLLM/issues)

이 페이지는 README 번역본입니다. 상세 가이드는 현재 영어로 제공됩니다. 문서 번역은 벤치마크 프롬프트, 데이터셋 또는 채점 방법을 변경하지 않습니다.

연구에 TrustLLM을 사용했다면 [ICML 2024 논문](https://openreview.net/forum?id=bWUU0LwwMp)을 인용해 주세요. 전체 BibTeX는 [CITATION.bib](CITATION.bib)에 있습니다. 코드는 [MIT 라이선스](LICENSE)를 따르며, 데이터셋에는 각 원 출처의 이용 조건이 적용됩니다.
