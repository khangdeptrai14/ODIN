import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_retrieval import (  # noqa: E402
    find_document_rank,
    find_first_rank,
    find_passage_rank,
    validate_answerable_question,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.passages = [
            {
                "passage_id": "DOC-001-0003",
                "filename": "arrays.pdf",
                "location": "page 5",
            },
            {
                "passage_id": "DOC-001-0003",
                "filename": "policy.docx",
                "location": "document",
            },
            {
                "passage_id": "DOC-003-0005",
                "filename": "sorting.pdf",
                "location": "page 5",
            },
        ]

    def test_find_first_rank_returns_first_matching_position(self) -> None:
        rank = find_first_rank(
            self.passages,
            lambda passage: passage["location"] == "page 5",
        )

        self.assertEqual(rank, 1)

    def test_passage_id_matching_uses_exact_filename(self) -> None:
        question = {
            "expected_filename": "policy.docx",
            "expected_passage_ids": ["DOC-001-0003"],
        }

        self.assertEqual(find_passage_rank(question, self.passages), 2)

    def test_location_matching_is_scoped_to_expected_filename(self) -> None:
        question = {
            "expected_filename": "sorting.pdf",
            "expected_locations": ["page 5"],
        }

        self.assertEqual(find_passage_rank(question, self.passages), 3)

    def test_correct_document_without_correct_passage_is_a_passage_miss(self) -> None:
        question = {
            "expected_filename": "arrays.pdf",
            "expected_locations": ["page 9"],
        }

        self.assertEqual(find_document_rank(question, self.passages), 1)
        self.assertIsNone(find_passage_rank(question, self.passages))

    def test_validation_rejects_answerable_question_without_passage_gold(
        self,
    ) -> None:
        question = {
            "id": "q001",
            "expected_filename": "policy.docx",
            "answerable": True,
        }

        with self.assertRaisesRegex(ValueError, "expected_passage_ids"):
            validate_answerable_question(question)

    def test_validation_accepts_passage_ids_or_locations(self) -> None:
        passage_id_question = {
            "id": "q001",
            "expected_filename": "policy.docx",
            "expected_passage_ids": ["DOC-001-0003"],
        }
        location_question = {
            "id": "q002",
            "expected_filename": "arrays.pdf",
            "expected_locations": ["page 5"],
        }

        validate_answerable_question(passage_id_question)
        validate_answerable_question(location_question)


if __name__ == "__main__":
    unittest.main()
