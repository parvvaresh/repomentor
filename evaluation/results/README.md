# Evaluation results

| File | What it is |
| --- | --- |
| `ablation-dev-lines.md` | Dev split, 60-line windows, plain chunk text (the v1 index). |
| `ablation-dev-treesitter.md` | Dev split, Tree-sitter chunks, plain chunk text. |
| `ablation-dev-treesitter-context.md` | Dev split, Tree-sitter chunks with file path and symbols in the embedded text (the current default), with and without the `ms-marco-MiniLM-L-6-v2` reranker. |
| `ablation-test-lines.md` | Held-out test split, v1 index, run once after choosing the defaults. |
| `ablation-test-treesitter-context.md` | Held-out test split, current default index, run once after choosing the defaults. |
| `judge_calibration.json` | Agreement and Cohen's kappa between the `qwen2.5:1.5b` judge and the human labels in `../generated_answers.json`. |
| `answers-v1-qwen2.5*.json`, `answers-v2-qwen2.5*.json` | The v1 and current answer pipelines on the answer gold set, both generated and judged with `qwen2.5:1.5b`, and their judge-based metrics. |
| `v1/` | The original results from the first version (16 questions, one repository, human-reviewed answers with `qwen3:1.7b`). |

Reproduce with the commands in the main README's *Evaluate* section. Decisions were made on dev; test was run once afterwards.
