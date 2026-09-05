import argparse
import os
import json
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
generator_opensource_path = os.path.join(current_dir, '..', 'generator', 'opensource')
generator_closedsource_path = os.path.join(current_dir, '..', 'generator', 'closedsource')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if generator_opensource_path not in sys.path:
    sys.path.insert(0, generator_opensource_path)
if generator_closedsource_path not in sys.path:
    sys.path.insert(0, generator_closedsource_path)

from retriever.unixcoder import retrieve_unixcoder
from utils.prompt import get_plan_user_prompt, PLAN_SYSTEM_PROMPT, build_final_user_prompt_from_plan
from utils.token import count_tokens, truncate_last_tokens


def _get_initial_ref_scores(sample: dict, candidate_references: List[Dict[str, Any]]) -> Dict[str, List[tuple]]:
    ref_map = {ref["id"]: ref.get("score") for ref in candidate_references}
    out: Dict[str, List[tuple]] = {}
    candidate_dict = sample.get("candidate") or {}
    for ctype in ["class", "function", "method", "variable", "segment"]:
        for cid, cand_info in (candidate_dict.get(ctype) or {}).items():
            refs = cand_info.get("references") or []
            ref_ids = [r if isinstance(r, str) else (r.get("id") if isinstance(r, dict) else None) for r in refs if r]
            ref_ids = [rid for rid in ref_ids if rid]
            if ref_ids:
                out[cid] = [(rid, ref_map.get(rid)) for rid in ref_ids]
    return out


def prune_references(sample: dict, candidate_references: List[Dict[str, Any]]) -> dict:
    candidate_ref_map = {ref["id"]: ref for ref in candidate_references if ref.get("component_type") in ["function", "method"]}

    if "candidate" not in sample:
        return sample
    candidate_dict = sample["candidate"]
    for cand_type in ["class", "function", "method", "variable", "segment"]:
        if cand_type not in candidate_dict:
            continue
        for cand_id, cand_info in candidate_dict[cand_type].items():
            references = cand_info.get("references") or []
            if not references:
                continue
            ref_ids = [r if isinstance(r, str) else (r.get("id") if isinstance(r, dict) else None) for r in references]
            ref_ids = [rid for rid in ref_ids if rid]
            refs_in_map = [candidate_ref_map[rid] for rid in ref_ids if rid in candidate_ref_map]
            best_ref_dict = max(refs_in_map, key=lambda r: (r.get("score") or 0)) if refs_in_map else None
            cand_info["references"] = [best_ref_dict] if best_ref_dict else []
    return sample


def retrieve_top_candidates(example: dict, retriever_type: str, top_k: int = 20, is_versionexec: bool = False) -> List[Dict[str, Any]]:
    if retriever_type == "unixcoder":
        lib_results, repo_results, _ = retrieve_unixcoder(example, top_k=None, context=["function", "class", "method", "variable"], external=True, extend=False, is_versionexec=is_versionexec)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    all_candidates = []
    if is_versionexec:
        for library, types in lib_results.items():
            for comp_type in ["function", "class", "method", "variable"]:
                if comp_type in types:
                    for comp_id, comp_info in types[comp_type].items():
                        all_candidates.append({
                            "id": comp_id,
                            "source_code": comp_info.get("source_code"),
                            "relative_path": comp_info.get("relative_path"),
                            "component_type": comp_type,
                            "score": comp_info.get("score"),
                            "library": comp_info.get("library"),
                            "is_third_party": True,
                        })
    else:
        for file_path, types in repo_results.items():
            for comp_type in ["function", "class", "method", "variable"]:
                if comp_type in types:
                    for comp_id, comp_info in types[comp_type].items():
                        all_candidates.append({
                            "id": comp_id,
                            "source_code": comp_info.get("source_code"),
                            "relative_path": comp_info.get("relative_path"),
                            "component_type": comp_type,
                            "score": comp_info.get("score"),
                            "library": None,
                            "is_third_party": False,
                        })
        for library, types in lib_results.items():
            for comp_type in ["function", "class", "method", "variable"]:
                if comp_type in types:
                    for comp_id, comp_info in types[comp_type].items():
                        all_candidates.append({
                            "id": comp_id,
                            "source_code": comp_info.get("source_code"),
                            "relative_path": comp_info.get("relative_path"),
                            "component_type": comp_type,
                            "score": comp_info.get("score"),
                            "library": comp_info.get("library") or library,
                            "is_third_party": True,
                        })
    all_candidates.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    return all_candidates


def _score_for_compare(score: Any) -> float:
    """Normalize score for comparison/sort: 'inf' (str) or float('inf') -> inf, else float(score)."""
    if score == "inf" or score == float("inf"):
        return float("inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("inf")


def build_predicted_components(sample: dict, implementation_plan: List[str]) -> List[Dict[str, Any]]:
    from utils.extract_entity import (
        extract_component_names_from_plan,
        match_components_from_plan,
        extract_target_function_id,
    )
    component_names = extract_component_names_from_plan(implementation_plan)
    target_function_ids = extract_target_function_id(sample) or set()
    if sample.get("full_name"):
        target_function_ids.add(sample["full_name"])
    matched_components, extended_refs = match_components_from_plan(
        component_names, sample["candidate"], implementation_plan, target_function_ids=target_function_ids
    )
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in matched_components:
        cid = c["component_id"]
        raw_score = c.get("score")
        entry = {
            "component_id": cid,
            "is_third_party": c.get("is_third_party"),
            "relative_path": c.get("relative_path"),
            "lib_name": c.get("library") if c.get("is_third_party") else None,
            "score": "inf" if (raw_score == float("inf") or raw_score == "inf") else raw_score,
            "source_code": c.get("source_code"),
        }
        if cid not in by_id or _score_for_compare(entry["score"]) > _score_for_compare(by_id[cid]["score"]):
            by_id[cid] = entry
    for r in extended_refs:
        cid = r["component_id"]
        raw_score = r.get("score")
        entry = {
            "component_id": cid,
            "is_third_party": r.get("is_third_party"),
            "relative_path": r.get("relative_path"),
            "lib_name": r.get("library") if r.get("is_third_party") else None,
            "score": "inf" if (raw_score == float("inf") or raw_score == "inf") else raw_score,
            "source_code": r.get("source_code"),
        }
        if cid not in by_id or _score_for_compare(entry["score"]) > _score_for_compare(by_id[cid]["score"]):
            by_id[cid] = entry
    return list(by_id.values())


def _first_docstring_start(source: str) -> int:
    best = -1
    for quote in ['"""', "'''"]:
        i = source.find(quote)
        if i == -1:
            continue
        start = i
        while start > 0 and source[start - 1] in 'rRuUbB':
            start -= 1
        if best == -1 or start < best:
            best = start
    return best


def _first_doc_line_from_cand(cand_info: dict) -> str:
    doc = (cand_info or {}).get('docstring') or (cand_info or {}).get('description') or ''
    if not doc or not isinstance(doc, str):
        return ''
    if doc.strip().upper() == 'DOCSTRING':
        return ''
    dot = doc.find('.')
    if dot >= 0:
        content = doc[: dot + 1]
    else:
        content = doc
    stripped = content.strip()
    if not stripped:
        return ''
    return stripped + ' ...'


def _append_doc_block(sketch: str, content: str, indent: str = '    ') -> str:
    """Append \"\"\" + content (each line indented) + \"\"\". No change if content empty."""
    if not content:
        return sketch
    lines = content.split('\n')
    body = '\n'.join(indent + line for line in lines)
    return sketch + '\n' + indent + '"""' + '\n' + body + '\n' + indent + '"""'


def get_sketch(source_code: str, component_type: str) -> str:
    import re
    
    if component_type in ('function', 'method'):
        doc_start = _first_docstring_start(source_code)
        if doc_start >= 0:
            return source_code[:doc_start].rstrip()
        lines = source_code.split('\n')
        sig_lines = []
        in_def = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                in_def = True
                sig_lines.append(line.rstrip())
                if '):' in stripped or ') ->' in stripped:
                    break
                continue
            if in_def:
                sig_lines.append(line.rstrip())
                if '):' in stripped or ') ->' in stripped:
                    break
        if sig_lines:
            return '\n'.join(sig_lines)
        return source_code.split('\n')[0].strip()
    
    elif component_type == 'variable':
        return source_code.strip() or source_code
    
    elif component_type == 'class':
        lines = source_code.split('\n')
        result_lines = []
        in_class = False
        indent_level = 0
        method_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('class '):
                in_class = True
                indent_level = len(line) - len(line.lstrip())
                result_lines.append(line.rstrip())
                continue
            if in_class:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and stripped:
                    if method_lines:
                        result_lines.extend(method_lines)
                        method_lines = []
                    break
                if stripped.startswith('def '):
                    if method_lines:
                        result_lines.extend(method_lines)
                        method_lines = []
                    method_lines = [line.rstrip()]
                    if '):' in stripped or ') ->' in stripped:
                        result_lines.extend(method_lines)
                        method_lines = []
                elif method_lines:
                    method_lines.append(line.rstrip())
                    if '):' in stripped or ') ->' in stripped:
                        result_lines.extend(method_lines)
                        method_lines = []
        if method_lines:
            result_lines.extend(method_lines)
        return '\n'.join(result_lines)
    
    return source_code


def _first_sentence_from_doc(text: str) -> str:
    """First sentence of docstring (split by '.'). Flatten newlines to space, strip. Returns '' if no text or placeholder DOCSTRING."""
    if not text or not isinstance(text, str):
        return ''
    flat = text.replace('\n', ' ').strip()
    if not flat or flat.upper() == 'DOCSTRING':
        return ''
    first = flat.split('.')[0].strip()
    if not first:
        return ''
    return first + '.'


def _signature_with_doc(cand_info: dict, comp_type: str) -> str:
    source = (cand_info or {}).get('source_code')
    sketch = get_sketch(source, comp_type) or source.split('\n')[0] or ''
    first_line = _first_doc_line_from_cand(cand_info)
    return _append_doc_block(sketch, first_line)


def _get_target_component_id(example: dict) -> Optional[str]:
    import re
    current_file = example.get('relative_path')
    if not current_file:
        return None
    task_type = example.get('type')
    if task_type == 'method':
        prompt = example.get('target_method_prompt') or ''
        m = re.search(r'def\s+(\w+)\s*\(', prompt)
        if not m:
            return None
        method_name = m.group(1)
        candidate = example.get('candidate')
        for comp_id in candidate.get('method'):
            if comp_id.endswith('@' + current_file) and comp_id.split('@')[0].endswith('.' + method_name):
                return comp_id
        return None
    prompt = example.get('target_function_prompt') or ''
    m = re.search(r'def\s+(\w+)\s*\(', prompt)
    if not m:
        return None
    name = m.group(1)
    return f"{name}@{current_file}"


def _collect_components_from_candidate(example: dict) -> List[tuple]:
    """Collect (comp_id, comp_type, relative_path, cand_info) for all candidates (repo and third-party, same behavior)."""
    candidate = example.get('candidate')
    out = []
    for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
        if comp_type not in candidate:
            continue
        for comp_id, cand_info in candidate[comp_type].items():
            rel_path = cand_info.get('relative_path')
            out.append((comp_id, comp_type, rel_path, cand_info))
    return out


def _load_dep_graph(repo_name: str, cache: dict, parser_dir) -> dict:
    if repo_name in cache:
        return cache[repo_name]
    path = Path(parser_dir) / repo_name / "dependency_graph.json"
    if not path.exists():
        cache[repo_name] = {}
        return {}
    with open(path, encoding="utf-8") as f:
        cache[repo_name] = json.load(f)
    return cache[repo_name]


def _load_versionexec_dep_graph(sample: dict, cache: dict, parser_dir) -> dict:
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
    loaded = []
    missing = []
    for lib_name in sorted(lib_names):
        path = Path(parser_dir) / lib_name / "dependency_graph.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                merged.update(json.load(f))
            loaded.append(lib_name)
        else:
            missing.append(lib_name)

    logger.info(
        f"[VersionExec dep_graph] loaded={loaded}  missing={missing}  total_nodes={len(merged)}"
    )
    cache[cache_key] = merged
    return merged


def _enrich_hybrid(entries: list, candidate: dict) -> list:
    result = []
    for entry in entries:
        comp_id = entry["id"]
        comp_type = entry["type"]
        comp_info = (candidate.get(comp_type) or {}).get(comp_id) or {}
        result.append({
            "id": comp_id,
            "type": comp_type,
            "relative_path": comp_info.get("relative_path"),
            "source_code": comp_info.get("source_code"),
            "is_third_party": comp_info.get("is_third_party", False),
            "lib_name": comp_info.get("library"),
            "is_non_used": entry.get("is_non_used", False),
            "score": entry["score"],
        })
    return result


def _enrich_candidates_with_graph(example: dict, graph: dict) -> dict:
    """Add outgoing_calls and base_classes from dependency_graph into each candidate (deep copy)."""
    import json
    example = json.loads(json.dumps(example))
    candidate = example.get('candidate')
    for comp_type in list(candidate.keys()):
        for comp_id, cand_info in candidate[comp_type].items():
            if comp_id not in graph:
                continue
            node = graph[comp_id]
            cand_info['outgoing_calls'] = node.get('outgoing_calls')
            if 'method' not in cand_info['outgoing_calls']:
                cand_info['outgoing_calls']['method'] = (node.get('outgoing_calls') or {}).get('method')
            if not cand_info.get('signature') and node.get('signature'):
                cand_info['signature'] = node['signature']
            if node.get('base_classes'):
                cand_info['base_classes'] = node['base_classes']
    return example


_TYPE_ORDER = {'class': 0, 'function': 1, 'variable': 2, 'segment': 3, 'method': 4}


def _build_sketch_signatures(example: dict, components: List[tuple], exclude_target: bool = False, top_k_ids: set = None, sketch_ids: set = None, score_map: Dict[str, float] = None) -> str:
    """Build repo sketch. top_k_ids components get full source_code; others get signature + first docstring line.
    Files are sorted ascending by avg score (lowest first, highest last → closest to target).
    #CURRENT_FILE is always pinned last."""
    top_k_ids = top_k_ids or set()
    score_map = score_map or {}
    current_file = example.get('relative_path')
    target_id = _get_target_component_id(example) if exclude_target else None
    current_items = [(c, t, p, i) for c, t, p, i in components if p == current_file]
    other_items = [(c, t, p, i) for c, t, p, i in components if p != current_file]
    parts = []
    by_file: Dict[str, List[tuple]] = {}
    for item in other_items:
        rel_path = item[2]
        if not rel_path:
            continue
        by_file.setdefault(rel_path, []).append(item)

    def _file_avg_score(rel_path: str) -> float:
        scores = [score_map.get(c, 0.0) for c, t, p, i in by_file[rel_path]]
        return sum(scores) / len(scores) if scores else 0.0

    for rel_path in sorted(by_file.keys(), key=_file_avg_score):
        items = sorted(by_file[rel_path], key=lambda x: (score_map.get(x[0], 0.0)))
        parts.append(f"#IMPORTED_FILE: {rel_path}")
        parts.append("")
        for comp_id, comp_type, _, cand_info in items:
            if comp_type == 'method':
                continue
            if exclude_target and comp_id == target_id:
                continue
            if comp_id in top_k_ids:
                parts.append(cand_info.get('source_code', '') or _signature_with_doc(cand_info, comp_type))
            else:
                parts.append(_signature_with_doc(cand_info, comp_type))
            parts.append("")
    if current_items:
        items = sorted(current_items, key=lambda x: (score_map.get(x[0], 0.0)))
        parts.append(f"#CURRENT_FILE: {current_file}")
        parts.append("")
        for comp_id, comp_type, _, cand_info in items:
            if comp_type == 'method':
                continue
            if exclude_target and comp_id == target_id:
                continue
            if comp_id in top_k_ids:
                parts.append(cand_info.get('source_code', '') or _signature_with_doc(cand_info, comp_type))
            else:
                parts.append(_signature_with_doc(cand_info, comp_type))
            parts.append("")
    return '\n'.join(parts).strip()


def _build_external_library_section(example: dict, top_k_ids: set = None, sketch_ids: set = None, score_map: Dict[str, float] = None, flat_rerank: bool = False) -> str:
    top_k_ids = top_k_ids or set()
    sketch_ids = sketch_ids or set()
    score_map = score_map or {}
    limit = bool(sketch_ids)
    candidate = example.get('candidate')

    all_comps = []
    for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
        if comp_type not in candidate:
            continue
        for comp_id, cand_info in candidate[comp_type].items():
            if not cand_info.get('is_third_party'):
                continue
            if limit and comp_id not in top_k_ids and comp_id not in sketch_ids:
                continue
            lib_name = cand_info.get('library') or cand_info.get('lib_name') or 'external'
            score = score_map.get(comp_id, 0.0)
            all_comps.append((score, comp_id, cand_info, comp_type, lib_name))

    if not all_comps:
        return ""

    lines = []

    if flat_rerank:
        for score, comp_id, cand_info, comp_type, lib_name in sorted(all_comps, key=lambda x: x[0]):
            lines.append(f"#LIB {lib_name}")
            if comp_id in top_k_ids:
                lines.append(cand_info.get('source_code', '') or _signature_with_doc(cand_info, comp_type))
            else:
                lines.append(_signature_with_doc(cand_info, comp_type))
            lines.append("")
    else:
        by_lib: Dict[str, List[tuple]] = {}
        for item in all_comps:
            by_lib.setdefault(item[4], []).append(item)

        def _lib_avg_score(lib_name: str) -> float:
            scores = [s for s, _, _, _, _ in by_lib[lib_name]]
            return sum(scores) / len(scores) if scores else 0.0

        for lib_name in sorted(by_lib.keys(), key=_lib_avg_score):
            lines.append(f"#{lib_name}")
            for score, comp_id, cand_info, comp_type, _ in sorted(by_lib[lib_name], key=lambda x: x[0]):
                if comp_id in top_k_ids:
                    lines.append(cand_info.get('source_code', '') or _signature_with_doc(cand_info, comp_type))
                else:
                    lines.append(_signature_with_doc(cand_info, comp_type))
                lines.append("")
            lines.append("")

    return '\n'.join(lines).strip()


def _build_repo_structure_section(example: dict, components: List[tuple], graph: Optional[Dict[str, Any]] = None) -> str:
    """Structure tree (imported files first, current file last). Uses utils.build_tree_structure.
    Plan prompt should use example['tree_structure'] from JSONL instead of calling this."""
    from utils.build_tree_structure import _build_sketch_tree as build_tree_from_utils
    return build_tree_from_utils(example, components, graph=graph, invokes_max_depth=5)


def build_sketch_prompt(example: dict, retrieved_candidates: List[Dict[str, Any]], graph: Optional[Dict[str, Any]] = None) -> tuple:
    """Build the three sections for plan prompt. Returns (external_information, repo_sketch, repo_structure).
    retrieved_candidates (top_k by similarity) get full source_code in the prompt; all others get signature only.
    Both sections sorted ascending by score (highest-score group last → closest to target function).
    repo_structure is always loaded from example['tree_structure'] (from JSONL); no build/fallback."""
    top_k_ids = {r['id'] for r in retrieved_candidates} if retrieved_candidates else set(example.get('_top_k_ids') or [])
    sketch_ids = set(example.get('_sketch_ids') or [])
    score_map: Dict[str, float] = {}
    candidate = example.get('candidate') or {}
    for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
        for comp_id, cand_info in (candidate.get(comp_type) or {}).items():
            raw = cand_info.get('score')
            if isinstance(raw, dict):
                score_map[comp_id] = float(raw.get('w_target', 0.0))
            elif raw is not None:
                try:
                    score_map[comp_id] = float(raw)
                except (TypeError, ValueError):
                    score_map[comp_id] = 0.0
    if graph:
        example = _enrich_candidates_with_graph(example, graph)
    components = _collect_components_from_candidate(example)
    is_versionexec = not example.get('relative_path')
    external_information = _build_external_library_section(example, top_k_ids, sketch_ids, score_map=score_map, flat_rerank=is_versionexec) or "(none)"
    repo_structure = example.get("tree_structure") if example.get("tree_structure") else ""
    if is_versionexec:
        repo_sketch = ""
    elif components:
        repo_sketch = _build_sketch_signatures(example, components, exclude_target=True, top_k_ids=top_k_ids, sketch_ids=sketch_ids, score_map=score_map)
    else:
        repo_sketch = "(no repo candidates)"
    return (external_information, repo_sketch, repo_structure)


def build_prompt_for_sample(example: dict, have_candidates: bool = True, graph: Optional[Dict[str, Any]] = None) -> str:
    task_type = example.get('type')
    if task_type == 'method':
        raw_target = example.get('target_method_prompt')
    else:
        raw_target = example.get('target_function_prompt')
    import_stmts = example.get('import_statements')
    target_prompt = ('\n'.join(import_stmts) + '\n\n' + raw_target) if import_stmts else raw_target
    if have_candidates:
        external_information, repo_sketch, repo_structure = build_sketch_prompt(example, [], graph=graph)
        return get_plan_user_prompt(external_information, repo_sketch, repo_structure, target_prompt, have_candidates=True)
    return get_plan_user_prompt("", "", "", target_prompt, have_candidates=False)

def generate_implementation_plan(example: dict,
                                 evaluator=None, backend: str = 'firework', model: str = None, have_candidates: bool = True, batch_size: int = 64, max_token: int = 4096) -> List[str]:
    user_prompt = build_prompt_for_sample(example, have_candidates=have_candidates)
    if max_token > 0 and count_tokens(user_prompt) > max_token:
        user_prompt = truncate_last_tokens(user_prompt, max_tokens=max_token)
    user_prompt = user_prompt + "\n\nBriefly reason about the implementation approach (6-10 sentences), then output the plan inside <OUTPUT></OUTPUT> tags as a JSON array."
    system_prompt = PLAN_SYSTEM_PROMPT
    
    if backend not in ['firework'] and evaluator is None:
        raise RuntimeError("Evaluator is required for generating plans.")
    if backend == 'firework':
        batch_size = 1

    batch_attempt = 0
    
    while True:
        batch_attempt += 1
        try:
            logger.info(f"Batch attempt {batch_attempt} - generating {batch_size} responses")
            
            if backend == 'gpt':
                responses = [evaluator.chat(user_prompt) for _ in range(batch_size)]
            elif backend == 'firework':
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generator'))
                from firework import generate
                
                batch_prompts = [user_prompt] * batch_size
                responses = generate(
                    prompts=batch_prompts,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=4096
                )
            else:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generator', 'opensource'))
                from generate import generate_batch_responses
                
                batch_prompts = [user_prompt] * batch_size
                responses = generate_batch_responses(
                    prompts=batch_prompts,
                    evaluator=evaluator,
                    max_tokens=4096,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    temperature=0.2,
                    num_return_sequences=1
                )
            
            logger.info(f"Generated {len(responses)} responses in batch {batch_attempt}")
            
            for idx, response in enumerate(responses):
                if not response or not response.strip():
                    continue
                
                if "<OUTPUT>" not in response:
                    continue
                
                try:
                    plan = parse_plan_response(response)
                    logger.info(f"✓ Found valid plan in response {idx+1}/{len(responses)} of batch {batch_attempt}")
                    logger.info(f"✓ Successfully parsed {len(plan)} steps!")
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"PARSED IMPLEMENTATION PLAN:")
                    logger.info(f"{'='*60}")
                    for i, step in enumerate(plan, 1):
                        print(f"Step {i}: {step}...")
                        print("\n")
                    logger.info(f"{'='*60}\n")
                    
                    return plan
                    
                except Exception:
                    continue
            
            logger.warning(f"✗ No valid plan found in batch {batch_attempt}, retrying with new batch...")
            continue
                
        except Exception as e:
            logger.error(f"Error in batch attempt {batch_attempt}: {e}")
            logger.warning("Retrying...")
            continue

def generate_implementation_plans_batch(samples_batch: List[dict],
                                        evaluator=None, backend: str = 'firework', model: str = None,
                                        have_candidates_batch: List[bool] = None, raw_responses_out: Optional[List[str]] = None, max_token: int = 4096) -> List[Optional[List[str]]]:
    system_prompt = PLAN_SYSTEM_PROMPT

    prompts = []
    for i, sample in enumerate(samples_batch):
        have_candidates = have_candidates_batch[i] if have_candidates_batch else bool(sample.get('candidate'))
        user_prompt = build_prompt_for_sample(sample, have_candidates=have_candidates)
        if max_token > 0 and count_tokens(user_prompt) > max_token:
            user_prompt = truncate_last_tokens(user_prompt, max_tokens=max_token)
        user_prompt = user_prompt + "\n\nBriefly reason about the implementation approach (2–3 sentences), then output the plan inside <OUTPUT></OUTPUT> tags as a JSON array."
        prompts.append(user_prompt)
    
    if backend == 'gpt':
        responses = [evaluator.chat(prompt) for prompt in prompts]
    elif backend == 'firework':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generator'))
        from firework import generate_batch

        responses = generate_batch(
            prompts=prompts,
            model=model,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=4096,
            batch_size=min(len(prompts), 10)
        )
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generator', 'opensource'))
        from generate import generate_batch_responses
        
        responses = generate_batch_responses(
            prompts=prompts,
            evaluator=evaluator,
            max_tokens=4096,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.2,
            num_return_sequences=1
        )
    
    if raw_responses_out is not None:
        raw_responses_out.extend(responses)
    
    plans = []
    for idx, response in enumerate(responses):
        if not response or not response.strip():
            plans.append(None)
            continue
        
        if "<OUTPUT>" not in response:
            plans.append(None)
            continue
        
        try:
            plan = parse_plan_response(response)
            plans.append(plan)
        except Exception as e:
            logger.warning(f"Failed to parse plan for sample {samples_batch[idx].get('id')}: {e}")
            plans.append(None)
    
    return plans


def parse_plan_response(response: str) -> List[str]:
    import re
    
    response = response.strip()
    
    match = re.search(r'<OUTPUT>\s*(.*?)\s*</OUTPUT>', response, re.DOTALL | re.IGNORECASE)
    if match:
        response = match.group(1).strip()
    
    if response.startswith('```python'):
        response = response.replace('```python', '').replace('```', '').strip()
    elif response.startswith('```json'):
        response = response.replace('```json', '').replace('```', '').strip()
    elif response.startswith('```'):
        response = response.replace('```', '').strip()
    
    if response.startswith('[') and response.endswith(']'):
        try:
            plan = eval(response)
            if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
                return plan
        except:
            pass
    
    lines = response.split('\n')
    plan = []
    for line in lines:
        line = line.strip()
        if line and (line.startswith('"') or line.startswith("'") or line.startswith('-') or (line and line[0].isdigit())):
            cleaned = line.lstrip('0123456789.-) ').strip('"\'",')
            if cleaned:
                plan.append(cleaned)
    
    if plan:
        return plan
    
    raise ValueError("Could not parse response into a list of steps")


def _write_task_log(
    log_handle,
    task_id: str,
    sample: dict,
    plan: List[str],
    raw_response: Optional[str],
    predicted_components: List[Dict[str, Any]],
    initial_ref_scores: Optional[Dict[str, List[tuple]]] = None,
    candidate_references: Optional[List[Dict[str, Any]]] = None,
    max_token: int = 0,
) -> None:
    from utils.extract_entity import (
        extract_component_names_from_plan,
        match_components_from_plan,
        extract_target_function_id,
    )

    have_candidates = bool(sample.get("candidate"))
    user_prompt = build_prompt_for_sample(sample, have_candidates=have_candidates)
    if max_token > 0 and count_tokens(user_prompt) > max_token:
        user_prompt = truncate_last_tokens(user_prompt, max_tokens=max_token)

    component_names = extract_component_names_from_plan(plan)
    target_function_ids = extract_target_function_id(sample) or set()
    if sample.get("full_name"):
        target_function_ids.add(sample["full_name"])
    matched_components, extended_refs = match_components_from_plan(
        component_names, sample["candidate"], plan, target_function_ids=target_function_ids
    )

    def _extended_id_for(cid: str) -> Optional[str]:
        for ctype in ["class", "function", "method", "variable", "segment"]:
            cand_info = ((sample.get("candidate") or {}).get(ctype) or {}).get(cid)
            if cand_info:
                refs = cand_info.get("references") or []
                if refs:
                    r = refs[0]
                    return r.get("id") if isinstance(r, dict) else r
        return None

    lines = [f"---{task_id}", "plan component_names (regex): " + ", ".join(sorted(component_names))]
    lines.append("matched_components: " + json.dumps([m["component_id"] for m in matched_components], ensure_ascii=False))
    lines.append("")
    lines.append("candidate_references (id, score, component_type, is_third_party):")
    if candidate_references:
        for ref in candidate_references:
            rid = ref.get("id", "")
            score = ref.get("score")
            ctype = ref.get("component_type", "")
            third = ref.get("is_third_party", False)
            lib = ref.get("library")
            lib_str = f" lib={lib}" if lib else ""
            lines.append(f"  {rid}  score={score}  type={ctype}  is_third_party={third}{lib_str}")
    else:
        lines.append("  (not available)")
    lines.append("")
    initial_ref_scores = initial_ref_scores or {}
    for m in matched_components:
        cid = m["component_id"]
        lines.append(f"component {cid}")
        ref_parts = []
        if cid in initial_ref_scores:
            for ref_id, score in initial_ref_scores[cid]:
                ref_parts.append(f"{ref_id} (score {score})")
        if ref_parts:
            lines.append("initial reference(s) for component " + cid + ": " + ", ".join(ref_parts))
        ext_id = _extended_id_for(cid)
        if ext_id:
            lines.append(f"-> extend component {ext_id}")
        lines.append("")
    lines.append("user_prompt (for plan):")
    lines.append(user_prompt)
    lines.append("")
    lines.append("raw_response LLM:")
    lines.append(raw_response or "(none)")
    lines.append("")
    lines.append("predicted_components: " + json.dumps([c.get("component_id") for c in predicted_components], ensure_ascii=False))
    lines.append("")

    log_handle.write("\n".join(lines))
    log_handle.flush()


def _append_plan_token(log_token_path: str, task_id, sample: dict, have_candidates: bool) -> None:
    try:
        from utils.token import count_tokens
    except ImportError:
        return
    user_prompt = build_prompt_for_sample(sample, have_candidates=have_candidates)
    try:
        num_tokens = count_tokens(user_prompt, encoding="cl100k_base")
    except Exception:
        return
    os.makedirs(os.path.dirname(log_token_path), exist_ok=True)
    with open(log_token_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"task_id": task_id, "plan_token": num_tokens}, ensure_ascii=False) + "\n")


def _append_plan_token_after(log_token_path: str, task_id, sample: dict, have_candidates: bool, max_token: int) -> None:
    try:
        from utils.token import count_tokens, truncate_last_tokens
    except ImportError:
        return
    user_prompt = build_prompt_for_sample(sample, have_candidates=have_candidates)
    if max_token > 0 and count_tokens(user_prompt) > max_token:
        user_prompt = truncate_last_tokens(user_prompt, max_tokens=max_token)
    try:
        num_tokens = count_tokens(user_prompt, encoding="cl100k_base")
    except Exception:
        return
    os.makedirs(os.path.dirname(log_token_path), exist_ok=True)
    with open(log_token_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"task_id": task_id, "plan_token": num_tokens}, ensure_ascii=False) + "\n")


def get_plan_done_ids(plan_output_file: str) -> set:
    """Return set of task_ids already present in the plan output file (for resume)."""
    done = set()
    if not os.path.isfile(plan_output_file):
        return done
    try:
        with open(plan_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    tid = obj.get('id')
                    if tid is not None:
                        done.add(str(tid))
    except Exception as e:
        logger.warning(f"Error reading plan file {plan_output_file}: {e}")
    return done


def process_benchmark(input_file: str, output_file: str, retriever_type: str,
                     evaluator=None, backend: str = 'firework', model: str = None, top_k: int = 20, is_versionexec: bool = False, batch_size: int = 64, log_file: Optional[str] = None, max_token: int = 4096,
                     hybrid_top_k: Optional[int] = None, parser_dir: Optional[str] = None):
    logger.info(f"Loading data from {input_file}")

    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples from input file")

    input_ids_ordered = [s.get('id') for s in samples]
    plan_done_ids = get_plan_done_ids(output_file)
    remaining_ids = [tid for tid in input_ids_ordered if tid is not None and str(tid) not in plan_done_ids]
    sample_by_id = {str(s.get('id')): s for s in samples if s.get('id') is not None}
    samples = [sample_by_id[str(tid)] for tid in remaining_ids]

    if plan_done_ids:
        logger.info(f"Found existing plan file {output_file}. Done: {len(plan_done_ids)} tasks. Remaining: {len(samples)}")
        file_mode = 'a'
    else:
        logger.info("No existing plan file found. Starting from beginning.")
        file_mode = 'w'

    hybrid_embedder = None
    dep_graph_cache = {}
    if hybrid_top_k and hybrid_top_k > 0:
        from retriever.hybrid_retriever import retrieve_hybrid
        from retriever.unixcoder import UniXcoderEmbedder
        from utils.extract_entity import extract_target_function_id
        hybrid_embedder = UniXcoderEmbedder()
        logger.info(f"Hybrid retriever enabled: top_k={hybrid_top_k}, parser_dir={parser_dir}")

    processed_count = 0
    failed_count = 0
    log_handle = open(log_file, 'a', encoding='utf-8') if log_file else None

    repo_root = Path(output_file).resolve().parents[2]
    log_token_path = os.path.join(repo_root, "data", "log_token", "plan_token.jsonl")
    log_token_path_after = os.path.join(repo_root, "data", "log_token", "plan_token_after.jsonl")

    with open(output_file, file_mode, encoding='utf-8') as f:
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating implementation plans"):
            batch = samples[i:i+batch_size]
            
            try:
                batch_samples = []
                batch_have_candidates = []
                batch_initial_ref_scores = []
                batch_candidate_references = []
                for sample in batch:
                    task_type = sample.get('type')

                    if task_type == 'method':
                        target_prompt = sample.get('target_method_prompt')
                    else:
                        target_prompt = sample.get('target_function_prompt')

                    if not target_prompt.strip():
                        logger.error(f"Empty target prompt for task {sample.get('id')}")
                        raise RuntimeError(f"Empty target prompt for task {sample.get('id')}")

                    candidate_references = retrieve_top_candidates(sample, retriever_type, top_k, is_versionexec)
                    _cand = sample.get('candidate') or {}
                    for _ref in candidate_references:
                        _ctype = _ref.get('component_type')
                        _cid = _ref['id']
                        _sc = _ref.get('score')
                        if _sc is not None and _ctype and _cid in (_cand.get(_ctype) or {}):
                            _cand[_ctype][_cid]['score'] = _sc
                    batch_initial_ref_scores.append(_get_initial_ref_scores(sample, candidate_references))
                    batch_candidate_references.append(candidate_references)
                    sample = prune_references(sample, candidate_references)
                    sample['_top_k_ids'] = [r['id'] for r in candidate_references[:top_k]]
                    if is_versionexec:
                        sample['_sketch_ids'] = [r['id'] for r in candidate_references[top_k:top_k + 50]]

                    _task_log_id = sample.get('id', '?')
                    if is_versionexec:
                        _n_total_cands = len(candidate_references)
                        _full_code_refs = candidate_references[:top_k]
                        _sketch_refs    = candidate_references[top_k:top_k + 50]
                        _n_skip = max(0, _n_total_cands - len(_full_code_refs) - len(_sketch_refs))

                        def _lib_break(refs):
                            cnt: dict = {}
                            for _r in refs:
                                _lib = _r.get("library") or "unknown"
                                cnt[_lib] = cnt.get(_lib, 0) + 1
                            return "  ".join(f"{_l}={_c}" for _l, _c in sorted(cnt.items()))

                        _full_lib_break = _lib_break(_full_code_refs)
                        _sig_lib_break  = _lib_break(_sketch_refs)
                        logger.info(
                            f"[Task {_task_log_id}] VersionExec plan context: "
                            f"total_candidates={_n_total_cands}  "
                            f"third_party_full_code={len(_full_code_refs)}  "
                            f"third_party_sig_only={len(_sketch_refs)}  "
                            f"skipped={_n_skip}  "
                            f"full_code_libs=({_full_lib_break})  "
                            f"sig_only_libs=({_sig_lib_break})"
                        )
                    else:
                        _n_total = len(candidate_references)
                        _n_full = min(top_k, _n_total)
                        _n_sig = max(0, _n_total - _n_full)
                        logger.info(
                            f"[Task {_task_log_id}] plan context: "
                            f"full_code={_n_full} (top_k by similarity)  "
                            f"signature_only={_n_sig}"
                        )

                    have_candidates = bool(sample.get('candidate'))
                    batch_samples.append(sample)
                    batch_have_candidates.append(have_candidates)

                raw_responses_out = [] if log_handle else None
                implementation_plans = generate_implementation_plans_batch(
                    batch_samples,
                    evaluator=evaluator,
                    backend=backend,
                    model=model,
                    have_candidates_batch=batch_have_candidates,
                    raw_responses_out=raw_responses_out,
                    max_token=max_token,
                )
                
                remaining_samples = []
                remaining_have_candidates = []
                
                successful_this_batch = 0
                for idx, plan in enumerate(implementation_plans):
                    if plan is None:
                        remaining_samples.append(batch_samples[idx])
                        remaining_have_candidates.append(batch_have_candidates[idx])
                    else:
                        batch_samples[idx]['implementation_plan'] = plan
                        if hybrid_top_k and hybrid_top_k > 0:
                            if is_versionexec:
                                _dep_graph = _load_versionexec_dep_graph(batch_samples[idx], dep_graph_cache, parser_dir)
                            else:
                                _dep_graph = _load_dep_graph(batch_samples[idx].get("repo_name", ""), dep_graph_cache, parser_dir)
                            _target_ids = extract_target_function_id(batch_samples[idx]) or set()
                            if batch_samples[idx].get("full_name"):
                                _target_ids.add(batch_samples[idx]["full_name"])
                            _retrieval_list = retrieve_hybrid(
                                batch_samples[idx], _dep_graph,
                                target_function_ids=_target_ids,
                                embedder=hybrid_embedder,
                                top_k=hybrid_top_k,
                                verbose=False,
                            )
                            batch_samples[idx]['predicted_components'] = _enrich_hybrid(_retrieval_list, batch_samples[idx].get("candidate") or {})
                        else:
                            batch_samples[idx]['predicted_components'] = build_predicted_components(
                                batch_samples[idx], plan
                            )
                        _append_plan_token(log_token_path, batch_samples[idx].get("id"), batch_samples[idx], batch_have_candidates[idx])
                        _append_plan_token_after(log_token_path_after, batch_samples[idx].get("id"), batch_samples[idx], batch_have_candidates[idx], max_token)
                        if log_handle:
                            raw_resp = raw_responses_out[idx] if raw_responses_out and idx < len(raw_responses_out) else None
                            _write_task_log(
                                log_handle,
                                str(batch_samples[idx].get("id", "")),
                                batch_samples[idx],
                                plan,
                                raw_resp,
                                batch_samples[idx]["predicted_components"],
                                batch_initial_ref_scores[idx],
                                candidate_references=batch_candidate_references[idx] if idx < len(batch_candidate_references) else None,
                                max_token=max_token,
                            )
                        batch_samples[idx].pop('_top_k_ids', None)
                        batch_samples[idx].pop('_sketch_ids', None)
                        json.dump(batch_samples[idx], f, ensure_ascii=False)
                        f.write('\n')
                        f.flush()

                        processed_count += 1
                        successful_this_batch += 1
                
                logger.info(f"Batch completed: {successful_this_batch}/{len(batch)} successful prompts")

                retry_count = 0
                while remaining_samples:
                    retry_count += 1
                    logger.info(f"Retry attempt {retry_count}: Regenerating {len(remaining_samples)} failed samples")

                    retry_raw_responses_out = [] if log_handle else None
                    retry_plans = generate_implementation_plans_batch(
                        remaining_samples,
                        evaluator=evaluator,
                        backend=backend,
                        model=model,
                        have_candidates_batch=remaining_have_candidates,
                        raw_responses_out=retry_raw_responses_out,
                        max_token=max_token,
                    )

                    new_remaining_samples = []
                    new_remaining_have_candidates = []

                    retry_successful = 0
                    for idx, plan in enumerate(retry_plans):
                        if plan is None:
                            new_remaining_samples.append(remaining_samples[idx])
                            new_remaining_have_candidates.append(remaining_have_candidates[idx])
                        else:
                            remaining_samples[idx]['implementation_plan'] = plan
                            if hybrid_top_k and hybrid_top_k > 0:
                                if is_versionexec:
                                    _dep_graph = _load_versionexec_dep_graph(remaining_samples[idx], dep_graph_cache, parser_dir)
                                else:
                                    _dep_graph = _load_dep_graph(remaining_samples[idx].get("repo_name", ""), dep_graph_cache, parser_dir)
                                _target_ids = extract_target_function_id(remaining_samples[idx]) or set()
                                if remaining_samples[idx].get("full_name"):
                                    _target_ids.add(remaining_samples[idx]["full_name"])
                                _retrieval_list = retrieve_hybrid(
                                    remaining_samples[idx], _dep_graph,
                                    target_function_ids=_target_ids,
                                    embedder=hybrid_embedder,
                                    top_k=hybrid_top_k,
                                    verbose=False,
                                )
                                remaining_samples[idx]['predicted_components'] = _enrich_hybrid(_retrieval_list, remaining_samples[idx].get("candidate") or {})
                            else:
                                remaining_samples[idx]['predicted_components'] = build_predicted_components(
                                    remaining_samples[idx], plan
                                )
                            _append_plan_token(log_token_path, remaining_samples[idx].get("id"), remaining_samples[idx], remaining_have_candidates[idx])
                            _append_plan_token_after(log_token_path_after, remaining_samples[idx].get("id"), remaining_samples[idx], remaining_have_candidates[idx], max_token)
                            if log_handle:
                                raw_resp = retry_raw_responses_out[idx] if retry_raw_responses_out and idx < len(retry_raw_responses_out) else None
                                _write_task_log(
                                    log_handle,
                                    str(remaining_samples[idx].get("id", "")),
                                    remaining_samples[idx],
                                    plan,
                                    raw_resp,
                                    remaining_samples[idx]["predicted_components"],
                                    candidate_references=None,
                                    max_token=max_token,
                                )
                            remaining_samples[idx].pop('_top_k_ids', None)
                            remaining_samples[idx].pop('_sketch_ids', None)
                            json.dump(remaining_samples[idx], f, ensure_ascii=False)
                            f.write('\n')
                            f.flush()

                            processed_count += 1
                            retry_successful += 1

                    logger.info(f"Retry attempt {retry_count}: {retry_successful}/{len(retry_plans)} successful, {len(new_remaining_samples)} remaining")

                    remaining_samples = new_remaining_samples
                    remaining_have_candidates = new_remaining_have_candidates
                
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                failed_count += len(batch)
                raise
    
    if log_handle:
        log_handle.close()
    logger.info(f"Successfully processed: {processed_count} samples")
    logger.info(f"Failed to process: {failed_count} samples")
    logger.info(f"Output saved to {output_file}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass
        if not os.getenv("FIREWORK_API_KEY"):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if v and (v.startswith('"') and v.endswith('"') or v.startswith("'") and v.endswith("'")):
                            v = v[1:-1]
                        if k:
                            os.environ.setdefault(k, v)

    parser = argparse.ArgumentParser(description="Generate implementation plans for benchmark tasks")
    parser.add_argument('--benchmark', choices=['RepoExec', 'DevEval', 'versionexec_old', 'versionexec_new'], 
                       required=True, help='Benchmark to process')
    parser.add_argument('--retriever', choices=['unixcoder'],
                       default='unixcoder', help='Retriever type to use')
    parser.add_argument('--model', type=str, required=True,
                       help="Direct model name. 'accounts/fireworks/models/' prefix is added automatically.")
    parser.add_argument('--top_k', type=int,
                       default=5, help='Number of top candidates to use for plan generation')
    parser.add_argument('--log', action='store_true',
                       help='Write per-task debug log to log.txt (retrieve, components, refs, extend, raw LLM response, predicted_components)')
    parser.add_argument('--max_token', type=int, default=4096,
                       help='Max tokens for plan user_prompt; if exceeded, truncate from the start (keep last max_token tokens). 0 = no limit (default: 4096)')
    parser.add_argument('--hybrid_top_k', type=int, default=5,
                       help='Top-K similarity candidates for hybrid retriever base nodes (regex-matched nodes always included). ')

    args = parser.parse_args()

    model = f'accounts/fireworks/models/{args.model}'

    input_file = os.path.join(str(repo_root), 'data', 'processed_benchmarks',
                             f'processed_{args.benchmark}_{args.model}.jsonl')
    if args.hybrid_top_k > 0:
        output_file = os.path.join(str(repo_root), 'data', 'processed_benchmarks',
                                   f'processed_{args.benchmark}_{args.model}_plan_hybrid.jsonl')
        parser_dir = str(repo_root / 'data' / 'parser_output' / args.benchmark)
    else:
        output_file = os.path.join(str(repo_root), 'data', 'processed_benchmarks',
                                   f'processed_{args.benchmark}_{args.model}_plan.jsonl')
        parser_dir = None

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run load_benchmark.py first to generate processed data")
        sys.exit(1)

    is_versionexec = args.benchmark in ['versionexec_old', 'versionexec_new']

    logger.info(f"Processing {args.benchmark} benchmark")
    logger.info(f"Retriever: {args.retriever}")
    logger.info(f"Model: {model}")
    logger.info(f"Top K: {args.top_k}")
    logger.info(f"Is VersionExec: {is_versionexec}")

    log_file = os.path.join(str(repo_root), "log.txt") if args.log else None
    if args.log:
        logger.info(f"Logging to {log_file}")
    process_benchmark(input_file, output_file, args.retriever, None, 'firework', model,
                      args.top_k, is_versionexec, batch_size=1,
                      log_file=log_file, max_token=args.max_token,
                      hybrid_top_k=args.hybrid_top_k, parser_dir=parser_dir)


if __name__ == "__main__":
    main()
