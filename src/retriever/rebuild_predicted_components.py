import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retriever.hybrid_retriever import retrieve_hybrid
from retriever.unixcoder import UniXcoderEmbedder
from utils.extract_entity import extract_target_function_id


def load_dep_graph(repo_name: str, cache: dict, parser_dir: Path) -> dict:
    if repo_name in cache:
        return cache[repo_name]
    path = parser_dir / repo_name / "dependency_graph.json"
    if not path.exists():
        cache[repo_name] = {}
        return {}
    with open(path, encoding="utf-8") as f:
        cache[repo_name] = json.load(f)
    return cache[repo_name]


def load_versionexec_dep_graph(sample: dict, cache: dict, parser_dir: Path) -> dict:
    candidate = sample.get("candidate") or {}
    lib_names = set()
    for comp_type_dict in candidate.values():
        for comp_info in comp_type_dict.values():
            lib = comp_info.get("lib_name") or comp_info.get("library")
            if lib:
                lib_names.add(lib)

    cache_key = tuple(sorted(lib_names))
    if cache_key in cache:
        return cache[cache_key]

    merged = {}
    for lib_name in lib_names:
        path = parser_dir / lib_name / "dependency_graph.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                merged.update(json.load(f))

    cache[cache_key] = merged
    return merged


def enrich(entries: list, candidate: dict, include_score: bool = True) -> list:
    result = []
    for entry in entries:
        comp_id = entry["id"]
        comp_type = entry["type"]
        comp_info = (candidate.get(comp_type) or {}).get(comp_id) or {}
        item = {
            "id": comp_id,
            "type": comp_type,
            "relative_path": comp_info.get("relative_path"),
            "source_code": comp_info.get("source_code"),
            "is_third_party": comp_info.get("is_third_party", False),
            "lib_name": comp_info.get("library"),
            "is_non_used": entry.get("is_non_used", False),
        }
        if include_score:
            item["score"] = entry["score"]
        result.append(item)
    return result


def get_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(str(json.loads(line).get("id")))
    return done


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild predicted_components using hybrid retriever.")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["RepoExec", "DevEval", "versionexec_old", "versionexec_new"],
        help="Benchmark name used to derive default input/output paths",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model tag used in filenames",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input JSONL file path (otherwise derived from --benchmark and --model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output JSONL file path (otherwise derived from --benchmark and --model)",
    )
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top-similarity (non-regex) candidates to include as base components")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed per-task retrieval logging")
    parser.add_argument("--id", nargs="+", default=None, metavar="ID",
                        help="Only process these task id values (e.g. --id 0 5 10)")
    parser.add_argument("--order", nargs="+", type=int, default=None, metavar="N",
                        help="Only process rows at these 0-based line numbers in the input file (e.g. --order 1 2 3)")
    args = parser.parse_args()

    if args.model is None and (args.input is None or args.output is None):
        parser.error("You must provide --model unless you pass both --input and --output explicitly")

    if args.input is None:
        args.input = REPO_ROOT / "data/processed_benchmarks" / f"processed_{args.benchmark}_{args.model}_plan_hybrid.jsonl"

    if args.output is None:
        args.output = REPO_ROOT / "data/processed_benchmarks" / f"processed_{args.benchmark}_{args.model}_plan_hybrid_final.jsonl"

    return args


def main():
    args = parse_args()

    parser_dir = REPO_ROOT / "data/parser_output" / args.benchmark

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.order is not None:
        order_set = set(args.order)
        samples = [s for i, s in enumerate(samples) if i in order_set]

    if args.id is not None:
        id_set = {str(i) for i in args.id}
        samples = [s for s in samples if str(s.get("id")) in id_set]

    done_ids = get_done_ids(args.output)
    remaining = [s for s in samples if str(s.get("id")) not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} done, {len(remaining)} remaining")

    print(f"top_k={args.top_k}  verbose={args.verbose}")

    is_versionexec = args.benchmark in ("versionexec_old", "versionexec_new")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    embedder = UniXcoderEmbedder()
    dep_graph_cache = {}

    with open(args.output, "a" if done_ids else "w", encoding="utf-8") as fout:
        for sample in tqdm(remaining, desc="Rebuilding predicted_components"):
            if is_versionexec:
                dep_graph = load_versionexec_dep_graph(sample, dep_graph_cache, parser_dir)
            else:
                repo_name = sample.get("repo_name", "")
                dep_graph = load_dep_graph(repo_name, dep_graph_cache, parser_dir)

            target_ids = extract_target_function_id(sample) or set()
            if sample.get("full_name"):
                target_ids.add(sample["full_name"])

            retrieval_list = retrieve_hybrid(
                sample, dep_graph,
                target_function_ids=target_ids,
                embedder=embedder,
                top_k=args.top_k,
                verbose=args.verbose,
            )

            sample["predicted_components"] = enrich(retrieval_list, sample.get("candidate") or {})

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
