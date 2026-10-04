"""Publish agent-readable copies from the same content as the HTML documentation."""

from pathlib import Path, PurePosixPath
from urllib.parse import urldefrag, urljoin

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


_pages = {}


def on_pre_build(config):
    # MkDocs serve may rebuild repeatedly in the same Python process.
    _pages.clear()


def on_page_context(context, page, config, nav):
    markdown_path = str(PurePosixPath(page.file.dest_uri).with_suffix(".md"))
    context["markdown_url"] = urljoin(config.site_url, markdown_path)
    context["source_schema"] = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "TrustLLM",
        "description": config.site_description,
        "url": config.site_url,
        "codeRepository": config.repo_url,
        "programmingLanguage": "Python",
        "license": "https://github.com/HowieHwong/TrustLLM/blob/main/LICENSE",
        "citation": "https://arxiv.org/abs/2401.05561",
    }
    _pages[page.file.src_uri] = {
        "page": page,
        "path": markdown_path,
        "url": context["markdown_url"],
        "order": nav.pages.index(page) if page in nav.pages else len(nav.pages),
    }
    return context


def on_post_build(config):
    """Export rendered articles, so includes, tables and examples stay synchronized."""
    site = Path(config.site_dir)
    urls = {}
    for entry in _pages.values():
        page = entry["page"]
        urls[page.canonical_url] = entry["url"]
        urls[urljoin(config.site_url, page.file.dest_uri)] = entry["url"]

    current, optional, full = [], [], []
    for entry in sorted(_pages.values(), key=lambda item: item["order"]):
        page = entry["page"]
        html = (site / page.file.dest_uri).read_text(encoding="utf-8")
        article = BeautifulSoup(html, "html.parser").select_one("article.md-content__inner")
        if article is None:
            raise ValueError(f"Missing documentation article: {page.file.src_uri}")
        for element in article.select(".headerlink, .doc-tools, .md-content__button, script"):
            element.decompose()
        for element in article.find_all(["a", "img"]):
            attribute = "href" if element.name == "a" else "src"
            if not element.get(attribute):
                continue
            absolute = urljoin(page.canonical_url, element[attribute])
            base, fragment = urldefrag(absolute)
            # Preserve anchors as links to HTML, where custom heading IDs exist.
            element[attribute] = absolute if fragment else urls.get(base, absolute)
        body = MarkdownConverter(heading_style="ATX", bullets="-").convert_soup(article).strip()
        description = page.meta.get("description", config.site_description)
        exported = f"Source: {page.canonical_url}\n\n{body}\n"
        destination = site / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(exported, encoding="utf-8")
        if page.meta.get("agent_index", True):
            link = f"- [{page.title}]({entry['url']}): {description}"
            (optional if page.meta.get("archived") else current).append(link)
            full.append(f"{exported}\n")

    intro = (
        "# TrustLLM\n\n"
        f"> {config.site_description}\n\n"
        "Research toolkit for the ICML 2024 TrustLLM benchmark. The maintained source "
        "workflow supports Python and CLI orchestration, local Hugging Face causal models, "
        "and text-only OpenAI-compatible Chat Completions endpoints. An AI agent can call "
        "this workflow; TrustLLM does not evaluate agent trajectories or tool use.\n\n"
        "Use the installation guide to install the current source version from GitHub. "
        "Generation completion reports are not benchmark scores. Scoring can "
        "require classifier downloads and paid judge or embedding APIs.\n\n"
    )
    index = intro + "## Documentation\n\n" + "\n".join(current)
    index += "\n\n## Optional\n\n" + "\n".join(optional)
    index += (
        f"\n- [Complete documentation]({config.site_url}llms-full.txt): All indexed pages in one file."
        f"\n- [Source repository]({config.repo_url}): Code, issues, examples and multilingual READMEs.\n"
    )
    (site / "llms.txt").write_text(index, encoding="utf-8")
    (site / "llms-full.txt").write_text(
        intro + "---\n\n" + "\n---\n\n".join(full), encoding="utf-8"
    )
