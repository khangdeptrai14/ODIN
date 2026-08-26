import csv
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

from bs4 import BeautifulSoup
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "passages.jsonl"
REGISTER_FILE = PROJECT_ROOT / "docs" / "data-register.csv"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
DOCX_WORD_NAMESPACE = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
)


def normalize_text(text: str) -> str:
    """Remove repeated whitespace while preserving readable text."""
    return re.sub(r"\s+", " ", text).strip()


def calculate_sha256(path: Path) -> str:
    """Calculate a fingerprint for the exact source file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def load_data_register(path: Path) -> dict:
    """Load the CSV register, supporting comma or semicolon delimiters."""
    if not path.exists():
        raise FileNotFoundError(f"Data register not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            reader = csv.DictReader(file, dialect=dialect)
        except csv.Error:
            delimiter = ";" if sample.partition("\n")[0].count(";") else ","
            reader = csv.DictReader(file, delimiter=delimiter)

        rows = list(reader)

    register = {}

    for row in rows:
        filename = row.get("filename", "").strip()

        if filename:
            register[filename] = row

    return register


def is_approved(record: dict) -> bool:
    """Check whether a document may enter the index."""
    permission = record.get("permission_status", "").strip().lower()
    status = record.get("status", "").strip().lower()
    personal_data = record.get("contains_personal_data", "").strip().lower()

    return (
        permission == "approved"
        and status == "active"
        and personal_data != "yes"
    )


def extract_sections(path: Path) -> list[dict]:
    """Extract text while preserving basic source locations."""
    extension = path.suffix.lower()
    sections = []

    if extension == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")

            if text:
                sections.append(
                    {
                        "location": f"page {page_number}",
                        "text": text,
                    }
                )

    elif extension == ".docx":
        with ZipFile(path) as archive:
            document = ElementTree.fromstring(
                archive.read("word/document.xml")
            )

        namespace = {"w": DOCX_WORD_NAMESPACE}
        paragraphs = []

        for paragraph in document.findall(".//w:p", namespace):
            text = normalize_text(
                "".join(
                    node.text or ""
                    for node in paragraph.findall(".//w:t", namespace)
                )
            )

            if text:
                paragraphs.append(text)

        if paragraphs:
            sections.append(
                {
                    "location": "document",
                    "text": " ".join(paragraphs),
                }
            )

    elif extension in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        content = soup.find("main") or soup.find("body") or soup
        text = normalize_text(content.get_text(" "))

        if text:
            sections.append(
                {
                    "location": "webpage",
                    "text": text,
                }
            )

    elif extension in {".txt", ".md"}:
        text = normalize_text(
            path.read_text(encoding="utf-8", errors="ignore")
        )

        if text:
            sections.append(
                {
                    "location": "document",
                    "text": text,
                }
            )

    return sections


def split_into_chunks(
    text: str,
    maximum_characters: int = 1000,
    overlap_characters: int = 150,
) -> list[str]:
    """Split text into overlapping, language-independent chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + maximum_characters, len(text))

        if end < len(text):
            possible_breaks = [
                text.rfind(separator, start, end)
                for separator in [".", "?", "!", "。", "？", "！"]
            ]

            best_break = max(possible_breaks)

            if best_break > start + maximum_characters // 2:
                end = best_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap_characters, start + 1)

    return chunks


def main() -> None:
    register = load_data_register(REGISTER_FILE)
    passages = []
    skipped_files = []

    source_files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in source_files:
        record = register.get(path.name)

        if record is None:
            skipped_files.append((path.name, "missing from data register"))
            continue

        if not is_approved(record):
            skipped_files.append((path.name, "not approved or not active"))
            continue

        file_hash = calculate_sha256(path)
        sections = extract_sections(path)
        passage_number = 1

        for section in sections:
            chunks = split_into_chunks(section["text"])

            for chunk in chunks:
                passage_id = (
                    f"{record.get('document_id', path.stem)}"
                    f"-{passage_number:04d}"
                )

                passages.append(
                    {
                        "passage_id": passage_id,
                        "document_id": record.get("document_id", ""),
                        "filename": path.name,
                        "title": record.get("title", path.stem),
                        "source_url": record.get("source_url", ""),
                        "source_owner": record.get("source_owner", ""),
                        "course": record.get("course", ""),
                        "location": section["location"],
                        "sha256": file_hash,
                        "text": chunk,
                    }
                )

                passage_number += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for passage in passages:
            file.write(json.dumps(passage, ensure_ascii=False) + "\n")

    print(f"Created {len(passages)} passages.")
    print(f"Output: {OUTPUT_FILE}")

    if skipped_files:
        print("\nSkipped files:")

        for filename, reason in skipped_files:
            print(f"  - {filename}: {reason}")


if __name__ == "__main__":
    main()