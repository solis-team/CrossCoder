import json
import os
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_JSONL = os.path.join(REPO_ROOT, "data", "processed_benchmarks", "processed_RepoExec_14B.jsonl")
GRAPH_DIR = os.path.join(REPO_ROOT, "data", "parser_output", "RepoExec", "python-string-utils")
DEPENDENCY_GRAPH_PATH = os.path.join(GRAPH_DIR, "dependency_graph.json")


def _first_sentence_from_doc(text: str) -> str:
    """First sentence of docstring (split by '.'). Flatten newlines to space, strip. Returns '' if no text or placeholder DOCSTRING."""
    if not text or not isinstance(text, str):
        return ""
    flat = text.replace("\n", " ").strip()
    if not flat or flat.upper() == "DOCSTRING":
        return ""
    first = flat.split(".")[0].strip()
    if not first:
        return ""
    return first + "."


def _component_display_name(comp_id: str, comp_type: str) -> str:
    """Display as 'name (type)' — strip part from @ onwards."""
    name = comp_id.split("@")[0] if "@" in comp_id else comp_id
    return f"{name} ({comp_type})"


def _component_display_with_doc(comp_id: str, comp_type: str, cand_info: dict) -> str:
    return _component_display_name(comp_id, comp_type)


def _path_from_comp_id(comp_id: str) -> str:
    """Extract relative path from comp_id (name@path)."""
    return comp_id.split("@", 1)[1] if "@" in comp_id else ""


def _get_target_component_id(example: dict) -> Optional[str]:
    """Infer target component id from target_function_prompt or target_method_prompt + relative_path."""
    import re
    current_file = example.get("relative_path", "")
    if not current_file:
        return None
    task_type = example.get("type", "function")
    if task_type == "method":
        prompt = example.get("target_method_prompt") 
        m = re.search(r"def\s+(\w+)\s*\(", prompt)
        if not m:
            return None
        method_name = m.group(1)
        candidate = example.get("candidate", {})
        for comp_id in candidate.get("method", {}):
            if comp_id.endswith("@" + current_file) and comp_id.split("@")[0].endswith("." + method_name):
                return comp_id
        return None
    prompt = example.get("target_function_prompt") 
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    if not m:
        return None
    name = m.group(1)
    return f"{name}@{current_file}"


def _collect_components_from_candidate(example: dict) -> List[Tuple[str, str, str, dict]]:
    candidate = example.get("candidate", {})
    out = []
    for comp_type in ["class", "function", "method", "variable", "segment"]:
        if comp_type not in candidate:
            continue
        for comp_id, cand_info in candidate[comp_type].items():
            if cand_info.get("is_third_party", False):
                continue
            rel_path = cand_info.get("relative_path", "")
            out.append((comp_id, comp_type, rel_path, cand_info))
    return out


def _method_parent_class_id(comp_id: str, rel_path: str) -> Optional[str]:
    if "@" not in comp_id:
        return None
    name_part, path_part = comp_id.rsplit("@", 1)
    if "." not in name_part:
        return None
    class_name = name_part.rsplit(".", 1)[0]
    return f"{class_name}@{path_part}"


def _build_id_to_info(
    components: List[Tuple[str, str, str, dict]], graph: dict
) -> Dict[str, Tuple[str, dict]]:
    id_to_info: Dict[str, Tuple[str, dict]] = {}
    for comp_id, comp_type, _rel_path, cand_info in components:
        id_to_info[comp_id] = (comp_type, cand_info)
    for comp_id, node in graph.items():
        if comp_id not in id_to_info:
            id_to_info[comp_id] = (node.get("component_type", "function"), node)
    return id_to_info


def _group_file_components(
    file_items: List[Tuple[str, str, str, dict]]
) -> Tuple[List[Tuple[str, str, str, dict]], Dict[str, List[Tuple[str, str, str, dict]]]]:
    """Split into top_level (class, function, variable, segment) and methods_by_class. Method belongs to class, not file."""
    top_level: List[Tuple[str, str, str, dict]] = []
    methods_by_class: Dict[str, List[Tuple[str, str, str, dict]]] = {}
    for item in file_items:
        comp_id, comp_type, rel_path, cand_info = item
        if comp_type == "method":
            parent_id = _method_parent_class_id(comp_id, rel_path)
            if parent_id is not None:
                methods_by_class.setdefault(parent_id, []).append(item)
            else:
                top_level.append(item)
        else:
            top_level.append(item)
    return top_level, methods_by_class


def _emit_component_children(
    lines: List[str],
    comp_id: str,
    comp_type: str,
    cand_info: dict,
    id_to_info: Dict[str, Tuple[str, dict]],
    prefix: str,
    visited: set,
    max_depth: int,
    caller_path: str,
    target_component_id: Optional[str] = None,
    placeholder_invokes: int = 3,
) -> None:
    """Emit inherits then invokes (recursive). Any component with comp_id == target_component_id shows only 3 placeholders (no real invokes) to avoid data leakage."""
    is_target = target_component_id is not None and comp_id == target_component_id
    num_placeholders = placeholder_invokes if is_target else 0
    if max_depth <= 0 and not (is_target and num_placeholders):
        return
    base_list = cand_info.get("base_classes", []) or cand_info.get("inherits", [])
    children = [(f"inherits {_component_display_with_doc(bid, id_to_info.get(bid, ('class', {}))[0], id_to_info.get(bid, (None, {}))[1])}", None, None, None, "") for bid in base_list]
    if comp_type != "class" and not is_target:
        outgoing = cand_info.get("outgoing_calls", {}) or {}
        callees = []
        for key in ("function", "method", "class", "variable"):
            callees.extend(outgoing.get(key, []))
        for callee_id in callees:
            callee_path = _path_from_comp_id(callee_id)
            ctype = "function"
            cinfo = {}
            if callee_id in id_to_info:
                ctype, cinfo = id_to_info[callee_id]
            children.append((f"invokes {_component_display_with_doc(callee_id, ctype, cinfo)}", callee_id, ctype, cinfo, callee_path or caller_path))
    for i, (label, cid, ctype, cinfo, callee_path) in enumerate(children):
        is_last = i == len(children) - 1 and num_placeholders <= 0
        branch = "└── " if is_last else "├── "
        cont = "    " if is_last else "│   "
        lines.append(prefix + branch + label)
        if cid is not None and cid not in visited and cinfo and max_depth > 1:
            visited.add(cid)
            _emit_component_children(lines, cid, ctype, cinfo, id_to_info, prefix + cont, visited, max_depth - 1, callee_path or caller_path, target_component_id, placeholder_invokes)
            visited.discard(cid)
    if is_target and num_placeholders > 0:
        placeholder_types = ["(function)", "(class)", "(variable)"]
        for i in range(min(num_placeholders, len(placeholder_types))):
            branch = "└── " if i == num_placeholders - 1 else "├── "
            lines.append(prefix + branch + "invokes ??? " + placeholder_types[i])


def _emit_file_tree(
    lines: List[str],
    path: str,
    top_level: List[Tuple[str, str, str, dict]],
    methods_by_class: Dict[str, List[Tuple[str, str, str, dict]]],
    id_to_info: Dict[str, Tuple[str, dict]],
    file_prefix: str,
    child_cont: str,
    invokes_max_depth: int,
    target_component_id: Optional[str] = None,
    current_file: str = "",
    placeholder_invokes: int = 3,
) -> None:
    """Emit tree for one file. Target component (task) is last; for target add placeholder_invokes x 'invokes ???'."""
    lines.append(file_prefix + path)
    if not top_level and not methods_by_class:
        return
    class_items = [t for t in top_level if t[1] == "class"]
    func_items = [t for t in top_level if t[1] == "function"]
    var_items = [t for t in top_level if t[1] == "variable"]
    seg_items = [t for t in top_level if t[1] == "segment"]
    ordered = class_items + func_items + var_items + seg_items
    if target_component_id and path == current_file:
        other = [t for t in ordered if t[0] != target_component_id]
        target_item = [t for t in ordered if t[0] == target_component_id]
        if target_item:
            ordered = other + target_item
        else:
            name_part = target_component_id.split("@")[0] if "@" in target_component_id else target_component_id
            task_type = "method" if "." in name_part else "function"
            synthetic = (target_component_id, task_type, path, {})
            ordered = other + [synthetic]
    for idx, (comp_id, comp_type, _rel_path, cand_info) in enumerate(ordered):
        is_last_sibling = idx == len(ordered) - 1
        branch = "└── " if is_last_sibling else "├── "
        cont = "    " if is_last_sibling else "│   "
        lines.append(child_cont + branch + "contains " + _component_display_with_doc(comp_id, comp_type, cand_info))
        sub_prefix = child_cont + cont
        if comp_type == "class":
            meth_list = methods_by_class.get(comp_id, [])
            for midx, (m_id, m_type, _, m_info) in enumerate(meth_list):
                m_last = midx == len(meth_list) - 1
                m_branch = "└── " if m_last else "├── "
                m_cont = "    " if m_last else "│   "
                lines.append(sub_prefix + m_branch + "contains " + _component_display_with_doc(m_id, m_type, m_info))
                _emit_component_children(lines, m_id, m_type, m_info, id_to_info, sub_prefix + m_cont, set(), invokes_max_depth, path, target_component_id, placeholder_invokes)
        _emit_component_children(lines, comp_id, comp_type, cand_info, id_to_info, sub_prefix, set(), invokes_max_depth, path, target_component_id, placeholder_invokes)


def _build_sketch_tree(
    example: dict,
    components: List[Tuple[str, str, str, dict]],
    graph: Optional[dict] = None,
    invokes_max_depth: int = 2,
) -> str:
    """Tree with │├└─; imported files first, current file last; method only under class.
    Class: inherits only. Function/method: invokes (recursive)."""
    graph = graph or {}
    current_file = example.get("relative_path", "")
    target_component_id = _get_target_component_id(example)
    id_to_info = _build_id_to_info(components, graph)
    by_file: Dict[str, List[Tuple[str, str, str, dict]]] = {}
    for item in components:
        comp_id, comp_type, rel_path, _ = item
        by_file.setdefault(rel_path, []).append(item)
    other_files = sorted(k for k in by_file if k != current_file)
    file_order = other_files + [current_file]
    lines: List[str] = []
    lines.append("")
    if file_order:
        lines.append(".")
    for idx, path in enumerate(file_order):
        if path not in by_file:
            continue
        file_items = by_file[path]
        top_level, methods_by_class = _group_file_components(file_items)
        is_last_file = idx == len(file_order) - 1
        file_prefix = "└── " if is_last_file else "├── "
        child_cont = "    " if is_last_file else "│   "
        _emit_file_tree(lines, path, top_level, methods_by_class, id_to_info, file_prefix, child_cont, invokes_max_depth, target_component_id, current_file, placeholder_invokes=3)
    return "\n".join(lines)


def load_first_sample(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    if not line.strip():
        raise SystemExit("Empty processed file")
    return json.loads(line)


def load_sample_by_id(path: str, sample_id) -> dict:
    """Load sample whose 'id' matches sample_id (int or str)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sid = sample.get("id")
            if sid == sample_id or str(sid) == str(sample_id):
                return sample
    raise SystemExit(f"Sample with id={sample_id!r} not found in {path}")


def load_dependency_graph(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_candidates_with_graph(example: dict, graph: dict) -> dict:
    """Add outgoing_calls (and signature if missing) from dependency_graph into each candidate."""
    example = json.loads(json.dumps(example))  
    candidate = example.get("candidate", {})
    for comp_type in list(candidate.keys()):
        for comp_id, cand_info in candidate[comp_type].items():
            if comp_id not in graph:
                continue
            node = graph[comp_id]
            cand_info["outgoing_calls"] = node.get("outgoing_calls", {})
            if "method" not in cand_info["outgoing_calls"]:
                cand_info["outgoing_calls"]["method"] = node.get("outgoing_calls", {}).get("method", [])
            if not cand_info.get("signature") and node.get("signature"):
                cand_info["signature"] = node["signature"]
            if node.get("base_classes"):
                cand_info["base_classes"] = node["base_classes"]
    return example


def build_tree_structure_for_sample(
    example: dict,
    graph: Optional[dict] = None,
    invokes_max_depth: int = 5,
) -> str:
    """Build full codebase structure tree (with invokes) for a sample. Uses graph for outgoing_calls.
    Call this from load_benchmark so processed samples have tree_structure; gen_plan can use it as-is."""
    graph = graph or {}
    if graph:
        example = enrich_candidates_with_graph(example, graph)
    components = _collect_components_from_candidate(example)
    return _build_sketch_tree(example, components, graph=graph, invokes_max_depth=invokes_max_depth)


def main(task_id=0):
    print("Loading task from:", PROCESSED_JSONL, "with id =", task_id)
    example = load_sample_by_id(PROCESSED_JSONL, task_id)
    print("  id:", example.get("id"))
    print("  relative_path (current_file):", example.get("relative_path"))
    print("  import_statements:", len(example.get("import_statements", [])))
    print()

    graph = {}
    if os.path.isfile(DEPENDENCY_GRAPH_PATH):
        print("Loading dependency_graph:", DEPENDENCY_GRAPH_PATH)
        graph = load_dependency_graph(DEPENDENCY_GRAPH_PATH)
        print("  components in graph:", len(graph))
        example = enrich_candidates_with_graph(example, graph)
        print("  enriched candidate with outgoing_calls from graph")
    else:
        print("No dependency_graph at", DEPENDENCY_GRAPH_PATH, "- using sample as-is")
    print()

    components = _collect_components_from_candidate(example)
    print("Collected components (is_third_party=False):", len(components))
    by_file = {}
    for comp_id, comp_type, rel_path, _ in components:
        by_file.setdefault(rel_path, []).append(comp_id)
    for path, ids in sorted(by_file.items()):
        print(f"  {path}: {len(ids)} components")
    print()

    tree_text = _build_sketch_tree(example, components, graph=graph, invokes_max_depth=5)
    print("=" * 60)
    print("STRUCTURE TREE OUTPUT")
    print("=" * 60)
    print(tree_text)
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test sketch structure tree")
    parser.add_argument("--id", type=str, default="0", help="Task/sample id to load (default: 0)")
    args = parser.parse_args()
    task_id = args.id
    try:
        task_id = int(task_id)
    except ValueError:
        pass
    main(task_id=task_id)
