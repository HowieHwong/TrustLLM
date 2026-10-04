<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">Artículo</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">Documentación</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">Datos</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">Clasificación</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <a href="README.md">English</a> &nbsp; / &nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp; / &nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp; / &nbsp;
  <a href="README.ja.md">日本語</a> &nbsp; / &nbsp;
  <a href="README.ko.md">한국어</a> &nbsp; / &nbsp;
  <strong>Español</strong> &nbsp; / &nbsp;
  <a href="README.fr.md">Français</a>
</p>

TrustLLM es un conjunto de herramientas de investigación de código abierto para evaluar la confiabilidad de los grandes modelos de lenguaje en **seis dimensiones**. Ejecuta el benchmark de ICML 2024 con pesos locales o una API de modelos, y conserva juntos los datos, la configuración y los resultados.

Para automatizar evaluaciones con un agente de IA, consulta la [guía de integración](docs/guides/agents.md) sobre CLI/Python y resultados JSON, o comienza por el [índice de documentación para agentes](https://howiehwong.github.io/TrustLLM/llms.txt).

## Primeros pasos

**1 — Instala el paquete.** El paquete básico permite generar respuestas mediante API y descargar los datos, sin bibliotecas de GPU.

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

Estos comandos instalan la **versión 0.4 directamente desde el código fuente en GitHub**. Se requiere Git. Para que tus experimentos sean reproducibles, sustituye `main` por el SHA de un commit concreto.

**2 — Descarga los datos del benchmark.**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — Prueba tu modelo mediante API.** Configura las credenciales y el endpoint del servicio:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

Sustituye la URL de ejemplo por la URL base real de la API de tu proveedor. Para un servidor local compatible con OpenAI, puedes usar `http://localhost:8000/v1` y el ID del modelo que sirve. La clave API es opcional si el servidor no exige autenticación. El modo API utiliza Chat Completions solo de texto.

Abre `runs/api-safety/report.html` para consultar el estado de la ejecución. `--limit 5` selecciona los cinco primeros ejemplos **de cada archivo de datos** para una prueba rápida. Para ejecutar el conjunto completo, elimina esa opción y utiliza un directorio de salida nuevo. Este informe muestra el progreso de la generación, no las puntuaciones del benchmark.

## Pesos locales, el mismo flujo

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

Los pesos de Hugging Face se descargan la primera vez que se usan. Para cargar pesos que ya tengas, sustituye el ID del modelo por `/path/to/checkpoint`. `--device cpu`, `cuda:0` y `mps` seleccionan el dispositivo; `auto` utiliza Accelerate para distribuir el modelo. La carga local admite modelos de lenguaje causales de Transformers y aplica la plantilla de chat del tokenizador cuando está disponible. Debes disponer del acceso al modelo, la capacidad de hardware y la compatibilidad de arquitectura necesarios.

## ¿Prefieres Python?

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

La configuración de la API se obtiene de `OPENAI_API_KEY` y `OPENAI_BASE_URL`, o de los argumentos explícitos `api_key` y `base_url` en Python. Consulta la [guía de uso y configuración](docs/guides/running.md) para obtener información sobre archivos JSON de configuración, reintentos, revisiones del modelo, ajustes de tokens y reanudación de ejecuciones.

## De las respuestas a las puntuaciones

Instala las dependencias de evaluación y evalúa un directorio que contenga todas las respuestas generadas:

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

Sustituye la ruta por el directorio de salida de una **ejecución completa**. Los resultados se guardan en `scores.json` y `scores.html`; las respuestas incompletas se rechazan. Los seis procesos de evaluación conservan los métodos del benchmark original. Según la tarea, la evaluación utiliza reglas, un clasificador descargado, embeddings o un modelo evaluador mediante API. Configura `OPENAI_JUDGE_MODEL` con un modelo evaluador al que tenga acceso tu cuenta. Estas llamadas pueden generar costes. Lee la [guía de evaluación](docs/guides/evaluation.md) antes de realizar una ejecución completa.

| Dimensión | Aspectos evaluados |
| :--- | :--- |
| **Veracidad** | Información falsa · alucinaciones · complacencia con el usuario |
| **Seguridad** | Ataques de jailbreak · uso indebido · comportamientos de seguridad excesivos |
| **Equidad** | Estereotipos · preferencias · menosprecio |
| **Robustez** | Perturbaciones adversarias · entradas fuera de distribución |
| **Privacidad** | Conciencia de privacidad · filtración de información |
| **Ética** | Juicios morales · decisiones morales |

[Referencia de conjuntos de datos y métricas →](docs/benchmark.md)

<details>
<summary><b>¿Qué cambia en la versión 0.4?</b></summary>

- IDs de modelo libres y un ejecutor compartido para la generación local y mediante API, sin la antigua lista de modelos permitidos.
- Instalación ligera para API; dependencias opcionales `local`, `eval` y `legacy` para el motor archivado.
- Comandos `download`, `tasks`, `generate` y `evaluate`, también disponibles mediante `python -m trustllm`.
- Reintentos limitados para API, puntos de control por ejemplo, registro explícito de fallos y comprobación de compatibilidad con `--resume`.
- Archivos JSON de respuestas compatibles con los evaluadores existentes basados en el campo `res`.
- Registro de hashes de los datos, configuración, versiones de dependencias, información de consumo cuando el servicio la devuelve e informes HTML de generación.

El motor de generación original está archivado en `trustllm.generation.legacy`. El formato de entrada y las integraciones de modelos han cambiado: las nuevas ejecuciones no equivalen automáticamente a la configuración del artículo original. Consulta las [notas de migración](docs/guides/running.md#migration-from-03).

</details>

## Investigación y desarrollo

[Comprobaciones de CI](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [Contribuir](CONTRIBUTING.md) · [Referencias de diseño](docs/design.md) · [Historial de cambios](docs/changelog.md) · [Incidencias](https://github.com/HowieHwong/TrustLLM/issues)

Esta página es una traducción del README. Las guías detalladas están disponibles actualmente en inglés. Traducir la documentación no modifica los prompts, los conjuntos de datos ni los métodos de evaluación del benchmark.

Si utilizas TrustLLM en tu investigación, cita el [artículo de ICML 2024](https://openreview.net/forum?id=bWUU0LwwMp). La entrada BibTeX completa está en [CITATION.bib](CITATION.bib). El código se distribuye bajo la licencia [MIT](LICENSE). Los conjuntos de datos conservan las condiciones de sus fuentes originales.
