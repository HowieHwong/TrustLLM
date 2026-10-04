"""Check the built docs and agent exports without network access.

Run after `python -m mkdocs build --strict` from the repository root.
"""

import json
import re
import textwrap
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

import yaml
from bs4 import BeautifulSoup


def check_site():
    config = yaml.safe_load(Path("mkdocs.yml").read_text(encoding="utf-8"))
    site = Path(config.get("site_dir", "site"))
    base = config["site_url"]
    errors = []
    documents = {}
    titles, descriptions = set(), set()

    def require(condition, message):
        if not condition:
            errors.append(message)

    def local_target(url):
        parsed = urlsplit(url)
        origin = urlsplit(base)
        if parsed.netloc != origin.netloc or not parsed.path.startswith(origin.path):
            return None
        relative = unquote(parsed.path[len(origin.path) :]) or "index.html"
        return site / relative, unquote(parsed.fragment)

    for file in site.rglob("*.html"):
        if file.name == "404.html":
            continue
        soup = BeautifulSoup(file.read_text(encoding="utf-8"), "html.parser")
        documents[file] = soup
        require(len(soup.select("article h1")) == 1, f"{file}: expected one article H1")
        title = soup.title.get_text() if soup.title else ""
        description = soup.find("meta", attrs={"name": "description"})
        description = description.get("content", "") if description else ""
        require(title and title not in titles, f"{file}: missing/duplicate title")
        require(
            description and description not in descriptions,
            f"{file}: missing/duplicate description",
        )
        titles.add(title)
        descriptions.add(description)
        canonical = soup.find("link", rel="canonical")
        expected_url = urljoin(base, file.relative_to(site).as_posix())
        require(canonical and canonical.get("href") == expected_url, f"{file}: wrong canonical URL")
        for property_name in ("og:title", "og:description", "og:url", "og:image"):
            tag = soup.find("meta", attrs={"property": property_name})
            require(tag and tag.get("content"), f"{file}: missing {property_name}")
        alternate = soup.find("link", rel="alternate", type="text/markdown")
        expected_markdown = urljoin(base, file.relative_to(site).with_suffix(".md").as_posix())
        require(
            alternate and alternate.get("href") == expected_markdown, f"{file}: wrong Markdown URL"
        )
        discovery = soup.find("link", rel="describedby")
        require(
            discovery and discovery.get("href") == base + "llms.txt",
            f"{file}: missing agent index link",
        )
        exported_file = file.with_suffix(".md")
        require(exported_file.is_file(), f"{file}: missing Markdown export")
        if exported_file.is_file():
            exported = exported_file.read_text(encoding="utf-8")
            require(f"Source: {expected_url}" in exported, f"{file}: wrong Markdown source URL")
            require(
                "--8<--" not in exported and "<div" not in exported, f"{file}: unprocessed export"
            )
            # The HTML is the authoritative article; code examples must survive export verbatim.
            # Markdown adds indentation when a fenced block is inside a list item.
            blocks = {
                textwrap.dedent(body).strip()
                for _, body in re.findall(
                    r"^([ \t]*)```[^\n]*\n(.*?)^\1```[ \t]*$", exported, re.M | re.S
                )
            }
            for code in soup.select("article pre code"):
                require(
                    textwrap.dedent(code.get_text()).strip() in blocks,
                    f"{file}: changed or missing exported code block",
                )

    for file, soup in documents.items():
        page_url = urljoin(base, file.relative_to(site).as_posix())
        for tag in soup.find_all(["a", "img", "link", "script"]):
            href = tag.get("href") or tag.get("src")
            if not href:
                continue
            target = local_target(urljoin(page_url, href))
            if target is None:
                continue
            path, fragment = target
            require(path.is_file(), f"{file}: missing local target {href}")
            if fragment and path in documents:
                require(
                    documents[path].find(id=fragment) is not None, f"{file}: missing anchor {href}"
                )

    index = (site / "llms.txt").read_text(encoding="utf-8")
    full = (site / "llms-full.txt").read_text(encoding="utf-8")
    require(index.startswith("# TrustLLM\n"), "llms.txt: missing project heading")
    require("## Optional" in index, "llms.txt: missing optional reference section")
    require("guides/agents.md" in index, "llms.txt: missing agent guide")
    for url in re.findall(r"\]\((https://[^)]+)\)", index):
        target = local_target(url)
        if target is None:
            continue
        path, _ = target
        require(path.is_file(), f"llms.txt: missing {url}")
        if path.suffix == ".md" and path.is_file():
            require(
                path.read_text(encoding="utf-8").strip() in full, f"llms-full.txt: missing {path}"
            )

    sitemap = ElementTree.parse(site / "sitemap.xml")
    locations = {element.text for element in sitemap.findall(".//{*}loc")}
    expected = {urljoin(base, file.relative_to(site).as_posix()) for file in documents}
    require(locations == expected, "sitemap.xml: document URLs do not match the built pages")
    schema = documents[site / "index.html"].find("script", type="application/ld+json")
    require(schema is not None, "Homepage: missing source-code structured data")
    if schema:
        data = json.loads(schema.string)
        require(
            data["@type"] == "SoftwareSourceCode" and data["codeRepository"] == config["repo_url"],
            "Homepage: incorrect structured data",
        )

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Validated {len(documents)} pages: metadata, local links/anchors, sitemap, Markdown examples and agent indexes."
    )


if __name__ == "__main__":
    check_site()
