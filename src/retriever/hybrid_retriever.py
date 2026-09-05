import json
import os
import sys
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

_retriever_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_retriever_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from tqdm import tqdm
from retriever.unixcoder import UniXcoderEmbedder, normalize_embedding
from sklearn.metrics.pairwise import cosine_similarity
from utils.extract_entity import extract_component_names_from_plan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.25
RETRIEVAL_TYPES = ["class", "function", "variable"]

_lib_embedding_cache: Dict[str, Dict[str, np.ndarray]] = {}


def _compute_w_target_scores(
    candidate: dict,
    target_prompt: str,
    embedder: UniXcoderEmbedder,
    is_versionexec: bool = False,
) -> Dict[str, float]:
    component_ids: List[str] = []
    texts: List[str] = []
    libs: List[Optional[str]] = []
    for comp_type in RETRIEVAL_TYPES:
        for comp_id, comp_info in (candidate.get(comp_type) or {}).items():
            component_ids.append(comp_id)
            texts.append(comp_info.get("source_code") or "")
            if is_versionexec and comp_info.get("is_third_party") and comp_info.get("library"):
                libs.append(comp_info.get("library"))
            else:
                libs.append(None)

    if not texts:
        return {}

    t_emb = normalize_embedding(embedder.get_embedding(target_prompt)[0])

    n = len(component_ids)
    normalized_embs: List[Optional[np.ndarray]] = [None] * n
    texts_to_compute: List[str] = []
    indices_to_compute: List[int] = []
    stash_keys: List[Optional[tuple]] = []

    cache_hits = 0
    cache_hit_libs: Dict[str, int] = {}
    for i, (cid, lib) in enumerate(zip(component_ids, libs)):
        if lib is not None:
            lib_bucket = _lib_embedding_cache.get(lib)
            if lib_bucket is not None and cid in lib_bucket:
                normalized_embs[i] = lib_bucket[cid]
                cache_hits += 1
                cache_hit_libs[lib] = cache_hit_libs.get(lib, 0) + 1
                continue
        texts_to_compute.append(texts[i])
        indices_to_compute.append(i)
        stash_keys.append((lib, cid) if lib is not None else None)

    newly_cached_libs: Dict[str, int] = {}
    first_seen_libs: set = set()
    if texts_to_compute:
        new_embs_raw = embedder.get_batch_embeddings(texts_to_compute, batch_size=64)
        for k, orig_i in enumerate(indices_to_compute):
            raw = new_embs_raw[k]
            norm = normalize_embedding(raw) if raw.ndim == 1 else normalize_embedding(raw[0])
            normalized_embs[orig_i] = norm
            stash = stash_keys[k]
            if stash is not None:
                lib, cid = stash
                if lib not in _lib_embedding_cache:
                    first_seen_libs.add(lib)
                _lib_embedding_cache.setdefault(lib, {})[cid] = norm
                newly_cached_libs[lib] = newly_cached_libs.get(lib, 0) + 1

    if newly_cached_libs:
        new_summary = ", ".join(
            f"{lib}(+{cnt}{'*' if lib in first_seen_libs else ''})"
            for lib, cnt in sorted(newly_cached_libs.items())
        )
        logger.info(f"[Hybrid] w_target cache: stored  libs=({new_summary})  (* = first time seen)")

    if cache_hits:
        hit_summary = ", ".join(f"{lib}({cnt})" for lib, cnt in sorted(cache_hit_libs.items()))
        logger.info(
            f"[Hybrid] w_target cache: hits={cache_hits}/{n}  computed={len(texts_to_compute)}  "
            f"libs=({hit_summary})"
        )

    c_embs_norm = np.vstack(normalized_embs)
    sims = cosine_similarity(t_emb.reshape(1, -1), c_embs_norm)[0]

    return {
        cid: float(sims[i])
        for i, cid in enumerate(component_ids)
    }


def _is_versionexec_test_component(comp_id: str) -> bool:
    return "test" in comp_id.lower()


def _compute_regex_flags(
    candidate: dict,
    implementation_plan: Optional[List[str]],
    target_function_ids: Set[str],
    is_versionexec: bool = False,
) -> Dict[str, int]:
    component_names = extract_component_names_from_plan(implementation_plan)
    flags: Dict[str, int] = {}

    for comp_type in RETRIEVAL_TYPES:
        for comp_id in (candidate.get(comp_type) or {}):
            if comp_id in target_function_ids:
                flags[comp_id] = 0
                continue
            if is_versionexec and _is_versionexec_test_component(comp_id):
                flags[comp_id] = 0
                continue
            comp_name = comp_id.split("@")[0]
            if is_versionexec:
                matched = any(comp_name == text for text in component_names)
            else:
                variants = (comp_name, comp_name.lower(), comp_name.upper())
                matched = any(v in text for text in component_names for v in variants)
            flags[comp_id] = 1 if matched else 0
    return flags


def _get_component_type(comp_id: str, candidate: dict) -> Optional[str]:
    for comp_type in RETRIEVAL_TYPES:
        if comp_id in (candidate.get(comp_type) or {}):
            return comp_type
    return None


def _select_base_components(
    w_target_scores: Dict[str, float],
    regex_flags: Dict[str, int],
    target_function_ids: Set[str],
    candidate: dict,
    top_k: int = 10,
    is_versionexec: bool = False,
) -> List[Dict[str, Any]]:
    regex_components = []
    non_regex_candidates = []
    for comp_id, w_t in w_target_scores.items():
        if comp_id in target_function_ids:
            continue
        if is_versionexec and _is_versionexec_test_component(comp_id):
            continue
        comp_type = _get_component_type(comp_id, candidate)
        if comp_type is None:
            continue
        regex = regex_flags.get(comp_id, 0)
        if regex == 1:
            regex_components.append({
                "id": comp_id,
                "base_score": 1.0,
                "type": comp_type,
            })
        else:
            non_regex_candidates.append((comp_id, w_t, comp_type))

    non_regex_candidates.sort(key=lambda x: x[1], reverse=True)
    topk_components = [
        {"id": cid, "base_score": w_t, "type": comp_type}
        for cid, w_t, comp_type in non_regex_candidates[:top_k]
    ]

    regex_ids = {bc["id"] for bc in regex_components}
    topk_components = [bc for bc in topk_components if bc["id"] not in regex_ids]

    return regex_components + topk_components


def _build_incoming_calls(
    candidate: dict,
    dependency_graph: dict,
) -> Dict[str, Set[str]]:
    candidate_ids: Set[str] = set()
    for comp_type in RETRIEVAL_TYPES:
        for comp_id in (candidate.get(comp_type) or {}):
            candidate_ids.add(comp_id)

    incoming_calls: Dict[str, Set[str]] = {cid: set() for cid in candidate_ids}
    for comp_id in candidate_ids:
        node = dependency_graph.get(comp_id)
        if not node:
            continue
        outgoing = node.get("outgoing_calls") or {}
        for call_type in RETRIEVAL_TYPES:
            for called_id in (outgoing.get(call_type) or []):
                if called_id in candidate_ids:
                    incoming_calls.setdefault(called_id, set()).add(comp_id)
    return incoming_calls


def _dfs_extend(
    current_node_id: str,
    base_score: float,
    hop: int,
    dependency_graph: dict,
    candidate: dict,
    incoming_calls: Dict[str, Set[str]],
    target_function_ids: Set[str],
    visited_in_branch: Set[str],
    is_versionexec: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    node = dependency_graph.get(current_node_id)

    outgoing_neighbors: Set[str] = set()
    if node:
        outgoing = node.get("outgoing_calls") or {}
        for call_type in RETRIEVAL_TYPES:
            for neighbor_id in (outgoing.get(call_type) or []):
                outgoing_neighbors.add(neighbor_id)

    incoming_neighbors: Set[str] = incoming_calls.get(current_node_id) or set()

    neighbors = []
    for neighbor_id in outgoing_neighbors | incoming_neighbors:
        if neighbor_id in target_function_ids:
            continue
        if neighbor_id in visited_in_branch:
            continue
        if is_versionexec and _is_versionexec_test_component(neighbor_id):
            continue
        if _get_component_type(neighbor_id, candidate) is None:
            continue
        neighbors.append(neighbor_id)

    for neighbor_id in neighbors:
        child_score = base_score / (hop + 1)
        comp_type = _get_component_type(neighbor_id, candidate)
        if child_score < SCORE_THRESHOLD:
            continue
        results.append({
            "child_id": neighbor_id,
            "score": child_score,
            "hop": hop,
            "parent_id": current_node_id,
            "type": comp_type,
        })
        if child_score >= 0.5:
            visited_in_branch.add(neighbor_id)
            deeper = _dfs_extend(
                neighbor_id,
                base_score,
                hop + 1,
                dependency_graph,
                candidate,
                incoming_calls,
                target_function_ids,
                visited_in_branch,
                is_versionexec=is_versionexec,
            )
            results.extend(deeper)
    return results


def _find_non_used_components(
    candidate: dict,
    dependency_graph: dict,
    target_function_ids: Set[str],
) -> Set[str]:
    if not dependency_graph:
        return set()

    all_outgoing_targets: Set[str] = set()
    for node_id, node in dependency_graph.items():
        if node_id in target_function_ids:
            continue
        outgoing = node.get("outgoing_calls") or {}
        for call_type in ["class", "function", "variable"]:
            for called_id in (outgoing.get(call_type) or []):
                all_outgoing_targets.add(called_id)

    non_used = set()
    for comp_type in RETRIEVAL_TYPES:
        for comp_id, comp_info in (candidate.get(comp_type) or {}).items():
            if comp_id in target_function_ids:
                continue
            if comp_info.get("is_third_party", False):
                continue
            if comp_id.split("@")[0].startswith("__"):
                continue
            if comp_id not in all_outgoing_targets:
                non_used.add(comp_id)

    return non_used


def retrieve_hybrid(
    sample: dict,
    dependency_graph: dict,
    target_function_ids: Optional[Set[str]] = None,
    embedder: Optional[UniXcoderEmbedder] = None,
    top_k: int = 10,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    candidate = sample.get("candidate") or {}
    implementation_plan = sample.get("implementation_plan")
    task_id = sample.get("id")

    if target_function_ids is None:
        from utils.extract_entity import extract_target_function_id
        target_function_ids = extract_target_function_id(sample) or set()
        if sample.get("full_name"):
            target_function_ids.add(sample["full_name"])

    task_type = sample.get("type")
    if task_type == "method":
        target_prompt = sample.get("target_method_prompt") or ""
    else:
        target_prompt = sample.get("target_function_prompt") or ""

    if embedder is None:
        embedder = UniXcoderEmbedder()

    is_versionexec = not sample.get("relative_path")

    w_target_scores = _compute_w_target_scores(candidate, target_prompt, embedder, is_versionexec=is_versionexec)
    regex_flags = _compute_regex_flags(candidate, implementation_plan, target_function_ids, is_versionexec=is_versionexec)

    for comp_type in RETRIEVAL_TYPES:
        for comp_id, comp_info in (candidate.get(comp_type) or {}).items():
            comp_info["score"] = {
                "w_target": w_target_scores.get(comp_id, 0.0),
                "regex": regex_flags.get(comp_id, 0),
            }

    retrieval_list: Dict[str, Dict[str, Any]] = {}

    base_components = _select_base_components(
        w_target_scores, regex_flags, target_function_ids, candidate, top_k=top_k, is_versionexec=is_versionexec,
    )

    for bc in base_components:
        cid = bc["id"]
        retrieval_list[cid] = {
            "id": cid,
            "score": bc["base_score"],
            "type": bc["type"],
        }

    best_by_child: Dict[str, Dict[str, Any]] = {}
    incoming_calls = _build_incoming_calls(candidate, dependency_graph)

    all_extended: List[Dict[str, Any]] = []
    for bc in base_components:
        visited = {bc["id"]}
        extended = _dfs_extend(
            bc["id"],
            bc["base_score"],
            1,
            dependency_graph,
            candidate,
            incoming_calls,
            target_function_ids,
            visited,
            is_versionexec=is_versionexec,
        )
        all_extended.extend(extended)

    for ext in all_extended:
        cid = ext["child_id"]
        if cid not in best_by_child or ext["score"] > best_by_child[cid]["score"]:
            best_by_child[cid] = ext

    for cid, ext in best_by_child.items():
        if cid in retrieval_list:
            if ext["score"] > retrieval_list[cid]["score"]:
                retrieval_list[cid]["score"] = ext["score"]
        else:
            retrieval_list[cid] = {
                "id": cid,
                "score": ext["score"],
                "type": ext["type"],
            }

    NON_USED_SCORE = 1.0
    non_used = _find_non_used_components(candidate, dependency_graph, target_function_ids)
    for cid in non_used:
        comp_type = _get_component_type(cid, candidate)
        if comp_type is None:
            continue
        if cid in retrieval_list:
            retrieval_list[cid]["score"] = max(retrieval_list[cid]["score"], NON_USED_SCORE)
        else:
            retrieval_list[cid] = {"id": cid, "score": NON_USED_SCORE, "type": comp_type, "is_non_used": True}

    result = sorted(retrieval_list.values(), key=lambda x: x["score"], reverse=True)

    if verbose:
        base_lines = "\n".join(f"  - {bc['id']} (score={bc['base_score']:.4f}, regex={regex_flags.get(bc['id'], 0)})" for bc in base_components) or "  (none)"
        logger.info(
            f"task {task_id}:\n"
            f"  target_function_id: {sorted(target_function_ids)}\n"
            f"  base_components:\n{base_lines}\n"
            f"  num_retrieved_component: {len(result)}"
        )

    n_third_party = sum(
        1 for r in result
        if (candidate.get(r["type"]) or {}).get(r["id"], {}).get("is_third_party", False)
    )
    n_repo = len(result) - n_third_party
    n_base_regex = sum(1 for bc in base_components if regex_flags.get(bc["id"], 0) == 1)
    n_base_sim = len(base_components) - n_base_regex

    logger.info(
        f"[Task {task_id}] total_retrieved={len(result)}  repo={n_repo}  third_party={n_third_party}  "
        f"base={len(base_components)}(regex={n_base_regex}, sim={n_base_sim})  "
        f"extended={len(best_by_child)}"
    )
    return result
