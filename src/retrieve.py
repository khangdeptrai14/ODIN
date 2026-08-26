import argparse
import json
import re
import textwrap
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PASSAGES_FILE = PROJECT_ROOT / "data" / "processed" / "passages.jsonl"


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and English text."""
    tokens = []

    for token in jieba.lcut(text.lower()):
        token = token.strip()

        if not token:
            continue

        if re.search(r"[a-z0-9_\u4e00-\u9fff]", token):
            tokens.append(token)

    return tokens


def load_passages(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run ingest.py first."
        )

    passages = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                passages.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    if not passages:
        raise ValueError("No passages were loaded. Check the data register.")

    return passages


def search(
    question: str,
    passages: list[dict],
    top_k: int = 5,
) -> list[tuple[float, dict]]:
    corpus_tokens = [tokenize(passage["text"]) for passage in passages]
    bm25 = BM25Okapi(corpus_tokens)

    question_tokens = tokenize(question)

    if not question_tokens:
        raise ValueError("The question did not contain searchable text.")

    scores = bm25.get_scores(question_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    return [
        (float(scores[index]), passages[index])
        for index in ranked_indices
    ]


def display_results(results: list[tuple[float, dict]]) -> None:
    for position, (score, passage) in enumerate(results, start=1):
        print(f"\nResult {position}")
        print(f"Score: {score:.4f}")
        print(f"Title: {passage.get('title', '')}")
        print(f"File: {passage.get('filename', '')}")
        print(f"Location: {passage.get('location', '')}")
        print(f"Passage ID: {passage.get('passage_id', '')}")

        source_url = passage.get("source_url", "")

        if source_url:
            print(f"Source: {source_url}")

        preview = textwrap.shorten(
            passage.get("text", ""),
            width=700,
            placeholder="...",
        )

        print(f"Text: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search ODIN's indexed course documents."
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question to search for",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to return",
    )

    arguments = parser.parse_args()
    question = arguments.question or input("Enter a question: ").strip()

    passages = load_passages(PASSAGES_FILE)
    results = search(question, passages, arguments.top_k)
    display_results(results)


if __name__ == "__main__":
    main()