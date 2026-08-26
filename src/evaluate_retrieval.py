import json
from collections.abc import Callable
from pathlib import Path

from retrieve import PASSAGES_FILE, load_passages, search


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
TOP_K = 5


def load_questions(path: Path) -> list[dict]:
    questions = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return questions


def validate_answerable_question(question: dict) -> None:
    question_id = question.get("id", "<unknown>")
    expected_filename = question.get("expected_filename")
    expected_passage_ids = question.get("expected_passage_ids")
    expected_locations = question.get("expected_locations")

    if not isinstance(expected_filename, str) or not expected_filename:
        raise ValueError(
            f"Answerable question {question_id} requires expected_filename."
        )

    has_passage_ids = (
        isinstance(expected_passage_ids, list)
        and bool(expected_passage_ids)
        and all(
            isinstance(passage_id, str) and passage_id
            for passage_id in expected_passage_ids
        )
    )
    has_locations = (
        isinstance(expected_locations, list)
        and bool(expected_locations)
        and all(
            isinstance(location, str) and location
            for location in expected_locations
        )
    )

    if not has_passage_ids and not has_locations:
        raise ValueError(
            f"Answerable question {question_id} requires a non-empty "
            "expected_passage_ids or expected_locations list."
        )


def find_first_rank(
    passages: list[dict],
    matches: Callable[[dict], bool],
) -> int | None:
    return next(
        (
            position
            for position, passage in enumerate(passages, start=1)
            if matches(passage)
        ),
        None,
    )


def find_document_rank(question: dict, passages: list[dict]) -> int | None:
    expected_filename = question["expected_filename"]
    return find_first_rank(
        passages,
        lambda passage: passage.get("filename") == expected_filename,
    )


def find_passage_rank(question: dict, passages: list[dict]) -> int | None:
    expected_filename = question["expected_filename"]
    expected_passage_ids = question.get("expected_passage_ids")

    if expected_passage_ids:
        passage_ids = set(expected_passage_ids)
        return find_first_rank(
            passages,
            lambda passage: (
                passage.get("filename") == expected_filename
                and passage.get("passage_id") in passage_ids
            ),
        )

    expected_locations = set(question["expected_locations"])
    return find_first_rank(
        passages,
        lambda passage: (
            passage.get("filename") == expected_filename
            and passage.get("location") in expected_locations
        ),
    )


def format_rank(rank: int | None) -> str:
    return f"rank {rank}" if rank is not None else "MISS"


def main() -> None:
    passages = load_passages(PASSAGES_FILE)
    questions = load_questions(QUESTIONS_FILE)

    answerable_questions = [
        question
        for question in questions
        if question.get("answerable", True)
    ]

    if not answerable_questions:
        raise ValueError("No answerable evaluation questions were found.")

    for question in answerable_questions:
        validate_answerable_question(question)

    document_hits = 0
    passage_hits = 0
    passage_reciprocal_rank_total = 0.0

    for question in answerable_questions:
        results = search(question["question"], passages, top_k=TOP_K)
        returned_passages = [passage for _, passage in results]
        document_rank = find_document_rank(question, returned_passages)
        passage_rank = find_passage_rank(question, returned_passages)

        if document_rank is not None:
            document_hits += 1

        if passage_rank is not None:
            passage_hits += 1
            passage_reciprocal_rank_total += 1 / passage_rank

        returned_metadata = [
            {
                "passage_id": passage.get("passage_id"),
                "filename": passage.get("filename"),
                "location": passage.get("location"),
            }
            for passage in returned_passages
        ]

        print(
            f"{question['id']}: document {format_rank(document_rank)}; "
            f"passage {format_rank(passage_rank)}"
        )
        print(f"  Question: {question['question']}")
        print(f"  Expected file: {question['expected_filename']}")

        if question.get("expected_passage_ids"):
            print(
                "  Expected passage IDs: "
                f"{question['expected_passage_ids']}"
            )
        else:
            print(
                "  Expected locations: "
                f"{question['expected_locations']}"
            )

        print(f"  Returned: {returned_metadata}")

    total = len(answerable_questions)

    print("\nEvaluation summary")
    print(f"Answerable questions: {total}")
    print(f"Correct document Recall@5: {document_hits / total:.2%}")
    print(f"Correct passage Recall@5: {passage_hits / total:.2%}")
    print(
        "Passage MRR@5: "
        f"{passage_reciprocal_rank_total / total:.4f}"
    )

    unanswerable_count = len(questions) - total
    print(f"Unanswerable questions recorded: {unanswerable_count}")


if __name__ == "__main__":
    main()
