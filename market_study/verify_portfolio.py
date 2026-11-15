from pathlib import Path
import re
import json

import pandas as pd

from market_study.publish_portfolio import (
    AUTHOR, EXPECTED_SOURCE_SHA256, SPLIT_DIGEST, EXPECTED_SELECTED_MODEL,
    FIGURES, TABLES, REPORTS, assert_evidence, generate_readme, load_json
)


def verify(root):
    preparation, training, evaluation, comparison, cv = assert_evidence(root)
    portfolio = root / "portfolio"
    manifest = load_json(portfolio / "evidence_manifest.json")
    if (
        manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or manifest.get("assignment_sha256") != SPLIT_DIGEST
        or manifest.get("selected_model") != EXPECTED_SELECTED_MODEL
        or manifest.get("author") != AUTHOR
    ):
        raise ValueError("Portfolio provenance mismatch")
    for folder, expected in (("figures", FIGURES), ("tables", TABLES), ("reports", REPORTS)):
        actual = {p.name for p in (portfolio / folder).iterdir() if p.is_file()}
        if actual != set(expected):
            raise ValueError(f"Unexpected {folder} portfolio contents: {actual ^ set(expected)}")
    if manifest.get("figures") != list(FIGURES) or manifest.get("tables") != list(TABLES) or manifest.get("reports") != list(REPORTS):
        raise ValueError("Curated manifest does not match release inventory")
    for png in (portfolio / "figures").glob("*.png"):
        if png.stat().st_size < 1000 or png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Damaged or unexpectedly small figure: {png.name}")
    for csv in (portfolio / "tables").glob("*.csv"):
        if pd.read_csv(csv).empty:
            raise ValueError(f"Empty reviewed table: {csv.name}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if readme != generate_readme(preparation, training, evaluation, comparison, cv):
        raise ValueError("README differs from evidence-generated version")
    if "not" not in readme.lower() or "geographic" not in readme.lower():
        raise ValueError("Generalization limitations missing")
    links = re.findall(r"!?(?:\[[^\]]+\])\(([^)]+)\)", readme)
    for link in links:
        if link.startswith(("http://", "https://")):
            continue
        if not (root / link).exists():
            raise ValueError(f"Broken local README link: {link}")
    books = sorted((root / "notebooks").glob("*.ipynb"))
    if len(books) != 3:
        raise ValueError("Expected three research notebooks")
    for book in books:
        notebook = json.loads(book.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not notebook.get("cells"):
            raise ValueError(f"Invalid notebook structure: {book}")
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(book), "exec")
                if cell.get("outputs"):
                    raise ValueError("Notebook output must not be checked in")
    return {"figures": len(FIGURES), "tables": len(TABLES), "reports": len(REPORTS), "notebooks": len(books), "links": sum(not link.startswith(("http://", "https://")) for link in links)}


def main():
    result = verify(Path.cwd())
    print("PASS - Source and assignment fingerprints verified")
    print("PASS - README matches genuine, held-out evaluation evidence")
    print("PASS - Local README links:", result["links"])
    print("PASS - Curated tables:", result["tables"], "figures:", result["figures"], "reports:", result["reports"])
    print("PASS - Notebook structure and Python cell syntax:", result["notebooks"])
    print("PASS - Portfolio release artifact validation")


if __name__ == "__main__":
    main()
