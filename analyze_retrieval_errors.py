from evaluate_ablation import (
    retrieve_expanded_hybrid,
    retrieve_semantic,
)
from evaluate_retrieval import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    score_case,
)
from rag_answer import load_index


TOP_K = 5


def get_files(
    question: str,
    index: dict,
    retrieval_function,
) -> list[str]:
    indices = retrieval_function(question, index, TOP_K)

    return [
        index["chunks"][index_value]["file_path"]
        for index_value in indices
    ]


def main() -> None:
    cases = load_dataset(DEFAULT_DATASET_PATH)
    index = load_index()

    difference_count = 0

    for case in cases:
        question = case["question"]
        expected = case["expected_files"]

        semantic_files = get_files(
            question,
            index,
            retrieve_semantic,
        )

        expanded_files = get_files(
            question,
            index,
            retrieve_expanded_hybrid,
        )

        semantic_score = score_case(semantic_files, expected)
        expanded_score = score_case(expanded_files, expected)

        same_result = (
            semantic_score["recall"] == expanded_score["recall"]
            and semantic_score["reciprocal_rank"]
            == expanded_score["reciprocal_rank"]
        )

        if same_result:
            continue

        difference_count += 1

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")

        print("\nSemantic:")
        for rank, file_path in enumerate(semantic_files, start=1):
            print(f"  {rank}. {file_path}")
        print(f"  Missing: {semantic_score['missing']}")

        print("\nExpanded hybrid:")
        for rank, file_path in enumerate(expanded_files, start=1):
            print(f"  {rank}. {file_path}")
        print(f"  Missing: {expanded_score['missing']}")

        print("-" * 70)

    print(
        f"\nQuestions with different results: "
        f"{difference_count}/{len(cases)}"
    )


if __name__ == "__main__":
    main()