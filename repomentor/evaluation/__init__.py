"""
Evaluation commands: python -m repomentor eval <command>

    prepare     clone the benchmark repositories and build their indexes
    retrieval   Hit@K / Recall@K / MRR@K for one retrieval method
    ablation    compare retrieval methods per split and question type
    errors      per-question differences between two methods
    answers     generate answers for the answer gold set
    review      label generated answers by hand
    judge       label answers with a model judge, or calibrate it
    score       answer-quality metrics from human or judge labels
    check       regression gate used in CI
"""

import argparse
from pathlib import Path

from .. import config
from ..retrieval import METHODS


def command_prepare(arguments) -> None:
    from .benchmarks import prepare
    from .retrieval import load_benchmarks

    for benchmark in load_benchmarks(arguments.benchmark):
        print(f"\n{benchmark['name']}")
        directory = prepare(benchmark, variant=arguments.variant)
        print(f"Index: {directory}")


def add_eval_commands(commands) -> None:
    from .ablation import command_ablation
    from .answers import BASELINE_PATH, DATASET_PATH, command_answers
    from .check import command_check
    from .errors import command_errors
    from .judge import command_judge
    from .retrieval import DEFAULT_K_VALUES, command_retrieval
    from .review import command_review
    from .score import command_score

    evaluation = commands.add_parser(
        "eval",
        help="Evaluate retrieval and answers.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = evaluation.add_subparsers(dest="eval_command", required=True)

    def benchmark_options(parser, split_default="all"):
        parser.add_argument("--benchmark", type=Path, action="append",
                            help="Benchmark JSON file (repeatable; default: all).")
        parser.add_argument("--split", choices=("dev", "test", "all"), default=split_default)

    prepare = sub.add_parser("prepare", help="Clone and index the benchmark repositories.")
    prepare.add_argument("--benchmark", type=Path, action="append")
    prepare.add_argument("--variant", default="", help="Index name suffix, e.g. 'lines'.")
    prepare.set_defaults(handler=command_prepare)

    retrieval = sub.add_parser("retrieval", help="Score one retrieval method.")
    benchmark_options(retrieval)
    retrieval.add_argument("--type", choices=("natural", "identifier"))
    retrieval.add_argument("--method", choices=METHODS, default=config.RETRIEVAL_METHOD)
    retrieval.add_argument("--top-k", type=int, default=config.TOP_K)
    retrieval.set_defaults(handler=command_retrieval)

    ablation = sub.add_parser("ablation", help="Compare retrieval methods.")
    benchmark_options(ablation, "dev")
    ablation.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    ablation.add_argument("--methods", nargs="+", choices=METHODS)
    ablation.add_argument("--rerank", help="Also evaluate every method with this cross-encoder.")
    ablation.add_argument("--variant", default="", help="Use the index built with this variant.")
    ablation.set_defaults(handler=command_ablation)

    errors = sub.add_parser("errors", help="Per-question differences between two methods.")
    benchmark_options(errors, "dev")
    errors.add_argument("--first", choices=METHODS, default="semantic")
    errors.add_argument("--second", choices=METHODS, default="expanded-hybrid")
    errors.add_argument("--top-k", type=int, default=config.TOP_K)
    errors.set_defaults(handler=command_errors)

    answers = sub.add_parser("answers", help="Generate answers for the answer gold set.")
    answers.add_argument("--data", type=Path, default=config.DATA_DIRECTORY)
    answers.add_argument("--dataset", type=Path, default=DATASET_PATH)
    answers.add_argument("--output", type=Path, default=Path("evaluation/runs/answers.json"))
    answers.add_argument("--top-k", type=int, default=config.TOP_K)
    answers.add_argument("--overwrite-baseline", action="store_true",
                         help=f"Allow writing to {BASELINE_PATH}.")
    answers.set_defaults(handler=command_answers)

    review = sub.add_parser("review", help="Label generated answers by hand.")
    review.add_argument("answers", type=Path)
    review.add_argument("--again", action="store_true", help="Review already reviewed answers too.")
    review.set_defaults(handler=command_review)

    judge = sub.add_parser("judge", help="Label answers with a model judge.")
    judge.add_argument("answers", type=Path)
    judge.add_argument("--model", help="Judge model (default: OLLAMA_JUDGE_MODEL).")
    judge.add_argument("--calibrate", action="store_true",
                       help="Compare with the human labels instead of saving judge labels.")
    judge.set_defaults(handler=command_judge)

    score = sub.add_parser("score", help="Answer metrics from labels.")
    score.add_argument("answers", type=Path, nargs="?", default=BASELINE_PATH)
    score.add_argument("--labels", choices=("manual_review", "judge_review"), default="manual_review")
    score.add_argument("--output", type=Path)
    score.set_defaults(handler=command_score)

    check = sub.add_parser("check", help="Fail when quality drops below thresholds.")
    check.add_argument("--retrieval", action="store_true")
    check.add_argument("--answers", action="store_true")
    check.set_defaults(handler=command_check)
