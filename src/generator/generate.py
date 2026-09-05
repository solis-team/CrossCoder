import argparse
import json
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tqdm import tqdm
from generator.firework import generate as fireworks_generate
from utils.token import count_tokens, truncate_last_tokens
from utils.prompt import CODEGEN_SYSTEM_PROMPT

BENCHMARK_TO_FILENAME = {
    "RepoExec": "repoexec.final.generated.jsonl",
    "DevEval": "deveval.final.generated.jsonl",
    "versionexec_old": "versionexec_old.final.generated.jsonl",
    "versionexec_new": "versionexec_new.final.generated.jsonl",
}

def default_prompt_source(repo_root: str, benchmark: str, model: str) -> str:
    if benchmark == "RepoExec":
        name = "RepoExec"
    elif benchmark == "DevEval":
        name = "DevEval"
    else:
        name = benchmark
    return os.path.join(repo_root, "data", "prompt", f"{name}_{model}_prompt_rerank.jsonl")


def default_output_dir(repo_root: str, benchmark: str, model: str) -> str:
    return os.path.join(repo_root, "data", "generation", f"{benchmark}_{model}")

DEFAULT_SYSTEM_PROMPT = CODEGEN_SYSTEM_PROMPT


def load_prompt_source(path: str):
    rows = []
    system_prompt = DEFAULT_SYSTEM_PROMPT
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "codegen-systemprompt" in obj:
                system_prompt = obj["codegen-systemprompt"] or DEFAULT_SYSTEM_PROMPT
                continue
            if "prompt" not in obj:
                continue
            rows.append(obj)
    return rows, system_prompt


def get_task_id(row: dict) -> str:
    if "id" in row and row["id"] is not None:
        return str(row["id"])
    if "task_id" in row and row["task_id"] is not None:
        return str(row["task_id"])
    raise KeyError("No id or task_id in row")


def _task_id_sort_key(row: dict) -> tuple:
    tid = row.get("task_id") or row.get("id")
    if tid is None:
        return (1, "")
    s = str(tid)
    if s.isdigit():
        return (0, int(s))
    return (0, s)


def normalize_task_id_for_output(tid) -> int | str:
    if tid is None:
        return tid
    s = str(tid)
    if s.isdigit():
        return int(s)
    return tid


def load_completed_ids(output_path: str) -> set:
    completed = set()
    if not os.path.isfile(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("task_id") if obj.get("task_id") is not None else obj.get("id")
                if tid is not None:
                    completed.add(str(tid))
            except json.JSONDecodeError:
                continue
    return completed


def run(
    prompt_source: str,
    benchmark: str,
    output_dir: str,
    model: str,
    num_sample: int = 3,
    max_token: int = 4096,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    repo_root: str = None,
    ids: list = None,
    order: list = None,
    concurrent_tasks: int = 2,
):
    if benchmark not in BENCHMARK_TO_FILENAME:
        raise ValueError(f"benchmark must be one of {list(BENCHMARK_TO_FILENAME)}")
    if not model:
        raise ValueError("--model is required")
    resolved_model = f"accounts/fireworks/models/{model}"
    generate = fireworks_generate
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parents[2])
    log_token_dir = os.path.join(repo_root, "data", "log_token")
    gen_token_path = os.path.join(log_token_dir, "gen_token.jsonl")
    gen_token_after_path = os.path.join(log_token_dir, "gen_token_after.jsonl")
    rows, system_prompt = load_prompt_source(prompt_source)
    if order is not None:
        order_set = set(order)
        rows = [row for i, row in enumerate(rows) if i in order_set]
    rows.sort(key=_task_id_sort_key)
    if ids is not None:
        ids_set = {str(i) for i in ids}
        rows = [row for row in rows if get_task_id(row) in ids_set]
    out_name = BENCHMARK_TO_FILENAME[benchmark]
    output_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_token_dir, exist_ok=True)
    completed_ids = load_completed_ids(output_path)
    rows = [r for r in rows if get_task_id(r) not in completed_ids]
    if completed_ids:
        print(f"Resume: {len(completed_ids)} already in {output_path}. Remaining: {len(rows)}")
    task_infos = []
    for row in rows:
        task_id = get_task_id(row)
        prompt = row.get("prompt") or ""
        if max_token > 0 and count_tokens(prompt) > max_token:
            prompt_send = truncate_last_tokens(prompt, max_tokens=max_token)
        else:
            prompt_send = prompt
        prompt_send = prompt_send + "\n\nYou should only reason briefly (around 5 sentences) and then write the code immediately. Avoid long reasoning.\n\nYOU MUST return EXACTLY ONE ```python ... ``` code block. The block MUST be self-contained and runnable: include ALL required `import` statements at the top of the block, then the complete solution function (signature + docstring + body). No other code blocks, no extra examples, no driver code, no usage demos — just imports + the single solution function inside one ```python ... ``` block. This is REQUIRED at all costs."
        task_infos.append({
            "task_id": task_id,
            "prompt": prompt,
            "prompt_send": prompt_send,
        })

    concurrent_tasks = max(1, int(concurrent_tasks))
    print(f"Concurrent tasks per batch: {concurrent_tasks}  (in-flight requests = {concurrent_tasks * num_sample})")

    pbar = tqdm(total=len(task_infos), desc=benchmark)
    for chunk_start in range(0, len(task_infos), concurrent_tasks):
        chunk = task_infos[chunk_start:chunk_start + concurrent_tasks]

        for ti in chunk:
            token_before = count_tokens(ti["prompt"])
            token_after = count_tokens(ti["prompt_send"])
            with open(gen_token_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"task_id": ti["task_id"], "gen_token": token_before}, ensure_ascii=False) + "\n")
            with open(gen_token_after_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"task_id": ti["task_id"], "gen_token": token_after}, ensure_ascii=False) + "\n")

        all_prompts = []
        for ti in chunk:
            all_prompts.extend([ti["prompt_send"]] * num_sample)
        raw_outputs = generate(
            prompts=all_prompts,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for i, ti in enumerate(chunk):
            start = i * num_sample
            solutions = [s if s is not None else "" for s in raw_outputs[start:start + num_sample]]
            non_empty = [s for s in solutions if s]
            while len(non_empty) < num_sample:
                need = num_sample - len(non_empty)
                extra = generate(
                    prompts=[ti["prompt_send"]] * need,
                    model=resolved_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for s in extra:
                    if s:
                        non_empty.append(s)
                    if len(non_empty) >= num_sample:
                        break
            response = non_empty[:num_sample]
            out_line = {
                "task_id": normalize_task_id_for_output(ti["task_id"]),
                "prompt": ti["prompt"],
                "response": response,
            }
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            pbar.update(1)
    pbar.close()
    print(f"Done. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--prompt_source",
        type=str,
        default=None,
        help="Prompt JSONL path (default: data/prompt/{benchmark}_{model}_prompt_rerank.jsonl)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["RepoExec", "DevEval", "versionexec_old", "versionexec_new"],
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/generation/{benchmark}_{model})",
    )
    parser.add_argument("--num_sample", type=int, default=3)
    parser.add_argument("--max_token", type=int, default=4096, help="Max tokens for prompt.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model", type=str, required=True,
                        help="Direct model name. 'accounts/fireworks/models/' prefix is added automatically.")
    parser.add_argument(
        "--ids", nargs="+", default=None, metavar="ID",
        help="Only generate for these task_id/id values (e.g. --ids 0 5 10)",
    )
    parser.add_argument(
        "--order", nargs="+", type=int, default=None, metavar="N",
        help="Only generate for rows at these 0-based line numbers in the prompt file (e.g. --order 1 2 3)",
    )
    parser.add_argument(
        "--concurrent_tasks", type=int, default=2,
        help="Number of tasks to process at the same time (default 2 -> %d req in flight)" % 2,
    )
    args = parser.parse_args()
    repo_root_str = str(repo_root)

    prompt_source = args.prompt_source or default_prompt_source(repo_root_str, args.benchmark, args.model)
    output_dir = args.output_dir or default_output_dir(repo_root_str, args.benchmark, args.model)
    run(
        prompt_source=prompt_source,
        benchmark=args.benchmark,
        output_dir=output_dir,
        model=args.model,
        num_sample=args.num_sample,
        max_token=args.max_token,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repo_root=repo_root_str,
        ids=args.ids,
        order=args.order,
        concurrent_tasks=args.concurrent_tasks,
    )


if __name__ == "__main__":
    main()
