"""Dependency-free, self-contained HTML run reports."""

import html
import json
from pathlib import Path


def write_report(run, path):
    escape = lambda value: html.escape(str(value), quote=True)
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{value['total']}</td><td>{value['successful']}</td><td>{value['failed']}</td><td>{value['pending']}</td></tr>"
        for name, value in run["files"].items()
    )
    model = run["settings"]["model"]
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrustLLM · Generation report</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#15263a;font:15px/1.6 system-ui,sans-serif}main{max-width:1050px;margin:55px auto;padding:0 25px}header{border-top:4px solid #16877f;padding-top:24px}.brand{font-size:13px;letter-spacing:3px;color:#16877f;font-weight:700}h1{font-size:42px;line-height:1.2;margin:12px 0}p{color:#566577}.cards{display:flex;gap:16px;margin:30px 0;flex-wrap:wrap}.card{flex:1;background:white;padding:22px;border:1px solid #dde4ec;border-radius:12px;min-width:150px}.card strong{display:block;font-size:30px}.label{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:1px}.table{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;margin:22px 0}th,td{text-align:left;padding:14px;border-bottom:1px solid #e4e9ef}th{background:#eaf0f5;font-size:12px;text-transform:uppercase}pre{padding:22px;background:#102439;color:#ddebf7;overflow:auto;border-radius:10px;font-size:12px}footer{font-size:12px;color:#637489;margin-top:35px}</style><main><header><div class="brand">TRUSTLLM / RUN REPORT</div>"""
    document += f"<h1>{escape(run['settings']['task'].title())}</h1><p>{escape(model['model'])} · {escape(model['backend'])} · {escape(run['status'])}</p></header>"
    document += (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><span class="label">{label}</span><strong>{escape(value)}</strong></div>'
            for label, value in (
                ("Samples", run["total"]),
                ("Successful", run["successful"]),
                ("Failed", run["failed"]),
            )
        )
        + "</div>"
    )
    document += "<p>This report measures generation completeness. It is not a trustworthiness score. Use the evaluation pipeline to score complete response files.</p>"
    document += f'<div class="table"><table><thead><tr><th>Dataset</th><th>Total</th><th>Successful</th><th>Failed</th><th>Pending</th></tr></thead><tbody>{rows}</tbody></table></div>'
    document += (
        "<details><summary>Run settings and environment</summary><pre>"
        + escape(json.dumps(run, indent=2, ensure_ascii=False))
        + "</pre></details>"
    )
    document += "<footer>TrustLLM · Text-only benchmark generation · Keep run.json and samples.jsonl with your results.</footer></main></html>"
    Path(path).write_text(document, encoding="utf-8")


def write_score_report(result, path):
    """Render original metric values without inventing an overall trust score."""
    values = []

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for name, item in value.items():
                flatten(item, f"{prefix} / {name}" if prefix else name)
        else:
            values.append((prefix, "Not evaluated" if value is None else value))

    flatten(result["scores"])
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(value))}</td></tr>"
        for name, value in values
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TrustLLM · Scores</title><style>
body{font:15px/1.6 system-ui,sans-serif;margin:0;background:#f3f5f8;color:#15263a}main{max-width:1000px;margin:55px auto;padding:0 25px}header{border-top:4px solid #16877f;padding-top:24px}.brand{font-size:13px;letter-spacing:3px;color:#16877f;font-weight:700}h1{font-size:42px}table{border-collapse:collapse;width:100%;background:#fff}td,th{text-align:left;padding:14px;border-bottom:1px solid #dde4ec}th{background:#eaf0f5}pre{padding:22px;background:#102439;color:#ddebf7;overflow:auto;border-radius:10px}p{color:#566577}</style><main><header><div class="brand">TRUSTLLM / EVALUATION</div>"""
    page += f"<h1>{html.escape(result['task'].title())} scores</h1></header>"
    page += "<p>Original scorer outputs. Metric scales and directions differ; these values are not combined into an overall score. Consult the benchmark metric reference before comparing runs.</p>"
    page += (
        f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
    )
    page += (
        "<details><summary>Inputs and environment</summary><pre>"
        + html.escape(json.dumps(result, indent=2, ensure_ascii=False))
        + "</pre></details></main></html>"
    )
    Path(path).write_text(page, encoding="utf-8")
