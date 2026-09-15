"""
Stage 1: Ingestion
Parse raw PDFs/HTML in data/raw/ into a list of {doc_id, source, text} dicts.

Provenance: name each file in data/raw/ to match a doc_id in data/sources.csv
(e.g. data/raw/learner_handbook.pdf <-> doc_id "learner_handbook" in sources.csv)
so url/date_accessed/driver_stage/topic get attached automatically. Files
with no matching row still ingest fine - they just won't carry provenance.
"""
import csv
import re
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SOURCES_CSV = Path(__file__).resolve().parent.parent / "data" / "sources.csv"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def parse_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return clean_text(soup.get_text(separator=" "))


def load_sources(path: Path = SOURCES_CSV) -> dict:
    """Return {doc_id: {url, date_accessed, driver_stage, topic}} from the provenance log."""
    lookup = {}
    if not path.exists():
        return lookup
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["doc_id"]] = row
    return lookup


def load_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Return list of {doc_id, source, text, url, date_accessed, driver_stage, topic}
    for every file in raw_dir. Provenance fields come from sources.csv when the
    filename stem matches a doc_id there; otherwise they're empty strings."""
    docs = []
    if not raw_dir.exists():
        return docs

    sources = load_sources()

    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            text = parse_pdf(path)
        elif path.suffix.lower() in (".html", ".htm"):
            text = parse_html(path)
        elif path.suffix.lower() == ".txt":
            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        else:
            continue

        if text:
            prov = sources.get(path.stem, {})
            docs.append({
                "doc_id": path.stem,
                "source": path.name,
                "text": text,
                "url": prov.get("url", ""),
                "date_accessed": prov.get("date_accessed", ""),
                # doc-level stage/topic from provenance log act as a fallback;
                # per-chunk tagging in chunking.py can still override this
                "doc_driver_stage": prov.get("driver_stage", ""),
                "doc_topic": prov.get("topic", ""),
            })

    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {RAW_DIR}")
    for d in docs[:3]:
        print(f"- {d['doc_id']} ({len(d['text'])} chars)")
