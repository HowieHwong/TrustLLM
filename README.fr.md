<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">Article</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">Documentation</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">Données</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">Classement</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <a href="README.md">English</a> &nbsp; / &nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp; / &nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp; / &nbsp;
  <a href="README.ja.md">日本語</a> &nbsp; / &nbsp;
  <a href="README.ko.md">한국어</a> &nbsp; / &nbsp;
  <a href="README.es.md">Español</a> &nbsp; / &nbsp;
  <strong>Français</strong>
</p>

TrustLLM est une boîte à outils de recherche open source qui évalue la fiabilité des grands modèles de langage selon **six dimensions**. Exécutez le benchmark d’ICML 2024 avec des poids locaux ou une API de modèles, et conservez ensemble les données, les paramètres et les résultats.

Pour automatiser les évaluations avec un agent IA, consultez le [guide d’intégration](docs/guides/agents.md) pour la CLI, Python et les résultats JSON, ou commencez par l’[index de documentation pour les agents](https://howiehwong.github.io/TrustLLM/llms.txt).

## Pour commencer

**1 — Installer.** Le paquet de base permet de générer des réponses via une API et de télécharger les données, sans bibliothèque GPU.

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

Ces commandes installent la **version 0.4 directement depuis le code source sur GitHub**. Git est requis. Pour assurer la reproductibilité des expériences, remplacez `main` par le SHA d’un commit précis.

**2 — Télécharger les données du benchmark.**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — Tester votre modèle via une API.** Configurez les identifiants et l’URL du service :

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

Remplacez l’URL d’exemple par l’URL de base réelle de l’API de votre fournisseur. Pour un serveur local compatible avec OpenAI, utilisez `http://localhost:8000/v1` et l’identifiant du modèle servi. La clé API est facultative si le serveur n’exige pas d’authentification. Le mode API utilise l’interface Chat Completions en texte seul.

Ouvrez `runs/api-safety/report.html` pour consulter l’état de l’exécution. `--limit 5` sélectionne les cinq premiers exemples **de chaque fichier de données** pour un test rapide. Pour une exécution complète, retirez cette option et choisissez un nouveau répertoire de sortie. Ce rapport décrit l’avancement de la génération, pas les scores du benchmark.

## Des poids locaux, la même procédure

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

Les poids hébergés sur Hugging Face sont téléchargés lors de la première utilisation. Pour charger des poids déjà disponibles, remplacez l’identifiant du modèle par `/path/to/checkpoint`. `--device cpu`, `cuda:0` et `mps` permettent de choisir le périphérique ; `auto` confie le placement du modèle à Accelerate. Le chargement local prend en charge les modèles de langage causaux de Transformers et applique le modèle de conversation du tokenizer lorsqu’il est disponible. Les droits d’accès au modèle, la capacité matérielle et la compatibilité de l’architecture restent nécessaires.

## Utiliser Python

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

La configuration de l’API provient de `OPENAI_API_KEY` et `OPENAI_BASE_URL`, ou des arguments Python explicites `api_key` et `base_url`. Consultez le [guide d’utilisation et de configuration](docs/guides/running.md) pour les fichiers JSON de configuration, les nouvelles tentatives, les révisions des modèles, les paramètres de tokens et la reprise d’une exécution interrompue.

## Des réponses aux scores

Installez les dépendances d’évaluation, puis évaluez un répertoire contenant toutes les réponses générées :

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

Remplacez le chemin par le répertoire de sortie de votre **exécution complète**. Les résultats sont enregistrés dans `scores.json` et `scores.html` ; les réponses incomplètes sont rejetées. Les six chaînes d’évaluation conservent les méthodes du benchmark original. Selon la tâche, l’évaluation utilise des règles, un classifieur téléchargé, des représentations vectorielles ou un modèle évaluateur via API. Définissez `OPENAI_JUDGE_MODEL` pour choisir un évaluateur accessible à votre compte. Ces appels peuvent entraîner des frais. Lisez le [guide d’évaluation](docs/guides/evaluation.md) avant une exécution complète.

| Dimension | Éléments évalués |
| :--- | :--- |
| **Véracité** | Informations erronées · hallucinations · complaisance envers l’utilisateur |
| **Sécurité** | Contournement des garde-fous (jailbreak) · usages abusifs · comportements de sécurité excessifs |
| **Équité** | Stéréotypes · préférences · dénigrement |
| **Robustesse** | Perturbations adversariales · entrées hors distribution |
| **Confidentialité** | Prise en compte de la confidentialité · fuites d’informations |
| **Éthique** | Jugements moraux · choix moraux |

[Référence des jeux de données et des métriques →](docs/benchmark.md)

<details>
<summary><b>Quelles sont les nouveautés de la version 0.4 ?</b></summary>

- Identifiants de modèles libres et moteur commun pour la génération locale et via API, sans l’ancienne liste de modèles autorisés.
- Installation légère pour les API ; dépendances facultatives `local`, `eval` et `legacy` pour le moteur archivé.
- Commandes `download`, `tasks`, `generate` et `evaluate`, également accessibles avec `python -m trustllm`.
- Nombre limité de nouvelles tentatives pour les API, points de reprise par exemple, échecs explicites et vérification de compatibilité avec `--resume`.
- Fichiers JSON de réponses compatibles avec les évaluateurs existants utilisant le champ `res`.
- Enregistrement des empreintes des données, des paramètres, des versions des dépendances, des informations de consommation renvoyées par le service et des rapports HTML de génération.

Le moteur de génération original est archivé dans `trustllm.generation.legacy`. Le format des entrées et les intégrations de modèles ont changé : les nouvelles exécutions ne correspondent pas automatiquement aux paramètres de l’article original. Consultez les [notes de migration](docs/guides/running.md#migration-from-03).

</details>

## Recherche et développement

[Vérifications CI](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [Contribuer](CONTRIBUTING.md) · [Références de conception](docs/design.md) · [Historique des modifications](docs/changelog.md) · [Signalement de problèmes](https://github.com/HowieHwong/TrustLLM/issues)

Cette page est une traduction du README. Les guides détaillés sont actuellement disponibles en anglais. La traduction de la documentation ne modifie ni les prompts, ni les jeux de données, ni les méthodes d’évaluation du benchmark.

Si vous utilisez TrustLLM dans vos recherches, veuillez citer l’[article d’ICML 2024](https://openreview.net/forum?id=bWUU0LwwMp). L’entrée BibTeX complète se trouve dans [CITATION.bib](CITATION.bib). Le code est distribué sous licence [MIT](LICENSE). Les jeux de données restent soumis aux conditions de leurs sources d’origine.
