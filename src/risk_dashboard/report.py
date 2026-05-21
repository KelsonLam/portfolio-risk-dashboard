"""Render a self-contained HTML report around the dashboard figure.

The output is a single .html file with the dashboard image and a metrics table
baked in (the image is embedded as base64, so the file stands alone and can be
emailed or opened anywhere). No web server, no external assets.
"""

from __future__ import annotations

import base64
from pathlib import Path


def _format_value(key: str, value: float) -> str:
    percent_like = (
        "return" in key.lower()
        or "volatility" in key.lower()
        or "drawdown" in key.lower()
        or "var" in key.lower()
    )
    if percent_like and "sharpe" not in key.lower():
        return f"{value * 100:,.2f}%"
    return f"{value:,.2f}"


def render_report(
    image_path: Path | str,
    metrics: dict[str, float],
    title: str = "Portfolio risk dashboard",
    out_path: Path | str = "results/dashboard.html",
) -> Path:
    """Write an HTML report embedding ``image_path`` and the metrics table."""
    image_path = Path(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

    rows = "\n".join(
        f"<tr><td>{key}</td><td class='num'>{_format_value(key, value)}</td></tr>"
        for key, value in metrics.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          margin: 2rem auto; max-width: 1100px; color: #1b1f24; }}
  h1 {{ font-size: 1.6rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
  td {{ border: 1px solid #d0d7de; padding: 0.4rem 0.9rem; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:nth-child(even) {{ background: #f6f8fa; }}
  img {{ max-width: 100%; border: 1px solid #d0d7de; border-radius: 6px; }}
  .note {{ color: #57606a; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<table>
{rows}
</table>
<img src="data:image/png;base64,{encoded}" alt="Risk dashboard">
<p class="note">VaR and CVaR are one-day figures at the stated confidence,
reported as positive losses. Volatility figures are annualized.</p>
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
