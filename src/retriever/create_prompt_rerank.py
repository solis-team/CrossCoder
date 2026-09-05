import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

_retriever_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_retriever_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.prompt import build_reranked_user_prompt_from_plan
from tqdm import tqdm


def _task_id_sort_key(tid) -> tuple:
    if tid is None:
        return (1, "")
    s = str(tid)
    if s.isdigit():
        return (0, int(s))
    return (0, s)


def get_prompt_done_ids(output_file: str) -> set:
    done = set()
    if not os.path.isfile(output_file):
        return done
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    tid = obj.get("id")
                    if tid is not None:
                        done.add(str(tid))
    except Exception:
        pass
    return done


def run_create_prompt_rerank(
    input_file: str,
    output_file: str,
    include_import: bool = True,
    threshold: float = 0.25,
    is_versionexec: bool = False,
) -> None:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not is_versionexec:
        samples.sort(key=lambda s: _task_id_sort_key(s.get("id")))

    done_ids = get_prompt_done_ids(output_file)
    samples = [s for s in samples if str(s.get("id", "")) not in done_ids]

    file_mode = "a" if done_ids else "w"
    if done_ids:
        print(f"Resuming: {len(done_ids)} already in output. Remaining: {len(samples)}")
    else:
        print(f"Starting from beginning. Total samples: {len(samples)}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, file_mode, encoding="utf-8") as f_out:
        for sample in tqdm(samples, desc="Creating reranked prompts"):
            prompt = build_reranked_user_prompt_from_plan(
                sample,
                include_import=include_import,
                threshold=threshold,
                is_versionexec=is_versionexec,
            )
            out = {
                "id": sample.get("id"),
                "prompt": prompt,
                "relative_path": sample.get("relative_path"),
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()
    print(f"Prompts saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create reranked code-gen prompts from plan JSONL. Each component gets its own tag, "
                    "ranked globally by score (ascending — highest score last, closest to target). "
                    "Third-party classes are always prepended before all other components."
    )
    parser.add_argument(
        "--benchmark",
        choices=["RepoExec", "DevEval", "versionexec_old", "versionexec_new"],
        required=True,
        help="Benchmark name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Direct model name used for file paths.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input plan JSONL (default: data/processed_benchmarks/processed_{benchmark}_{model}_plan_hybrid_final.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prompt JSONL (default: data/prompt/{benchmark}_{model}_prompt_rerank.jsonl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output file before running, forcing full regeneration from scratch.",
    )
    parser.add_argument(
        "--include_import",
        choices=["true", "false"],
        default="true",
        help="Whether to include import statements in the prompt (true/false).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Minimum score for a predicted component to be included in the prompt. "
             "Components with score < threshold are excluded. Default: 0.25 (same as hybrid retriever minimum).",
    )
    args = parser.parse_args()

    is_versionexec = args.benchmark in ("versionexec_old", "versionexec_new")

    repo_root = Path(__file__).resolve().parents[2]
    default_input = str(
        repo_root / "data" / "processed_benchmarks" / f"processed_{args.benchmark}_{args.model}_plan_hybrid_final.jsonl"
    )
    default_output = str(
        repo_root / "data" / "prompt" / f"{args.benchmark}_{args.model}_prompt_rerank.jsonl"
    )

    input_file = args.input or default_input
    output_file = args.output or default_output

    if args.force and os.path.isfile(output_file):
        os.remove(output_file)
        print(f"--force: deleted {output_file}")

    if not os.path.isfile(input_file):
        print(f"Error: input file not found: {input_file}")
        sys.exit(1)

    include_import = args.include_import == "true"

    run_create_prompt_rerank(
        input_file,
        output_file,
        include_import=include_import,
        threshold=args.threshold,
        is_versionexec=is_versionexec,
    )


if __name__ == "__main__":
    main()
