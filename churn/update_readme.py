"""Insert real, generated results into README.md between marker comments.

Run after `python -m churn.eda` and `python -m churn.train`:
    python -m churn.update_readme
"""
import re

from . import config

README = config.ROOT / "README.md"


def _replace(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pat.search(text):
        raise ValueError(f"Marker {tag} not found in README.md")
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def main():
    text = README.read_text(encoding="utf-8")
    metrics = (config.REPORTS_DIR / "metrics.md")
    if metrics.exists():
        body = metrics.read_text(encoding="utf-8").split("\n", 2)[2].strip()  # drop the H1 title
        text = _replace(text, "METRICS", body)
    figs = sorted(config.FIGURES_DIR.glob("*.png"))
    if figs:
        md = "\n\n".join(f"**{f.stem.replace('_', ' ')}**\n\n![{f.stem}](reports/figures/{f.name})" for f in figs)
        text = _replace(text, "FIGURES", md)
    README.write_text(text, encoding="utf-8")
    print("README.md updated with generated results.")


if __name__ == "__main__":
    main()
