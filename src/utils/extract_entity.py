
import re
from typing import Any, Dict, List, Optional, Set


def extract_component_names_from_plan(implementation_plan: Optional[List[str]]) -> List[str]:
    if not implementation_plan:
        return []

    pattern = r"[`']([^`']+)[`']"
    backtick_texts: List[str] = []
    for step in implementation_plan:
        backtick_texts.extend(re.findall(pattern, step))
    return backtick_texts


def _method_to_class_id(component_id: str) -> Optional[str]:
    """Return class_id for a method component_id (e.g. Request.copy@path -> Request@path), else None."""
    if "@" not in component_id or "." not in component_id.split("@")[0]:
        return None
    name, path = component_id.split("@", 1)
    class_name = name.split(".")[0]
    return f"{class_name}@{path}"


def match_components_from_plan(
    component_names: Set[str],
    candidate: Dict[str, Any],
    implementation_plan: Optional[List[str]] = None,
    target_function_ids: Optional[Set[str]] = None,
) -> tuple:
    """Match component_names to candidate dict; return (matched_components, extended_refs)."""
    matched_components: List[Dict[str, Any]] = []
    extended_refs: List[Dict[str, Any]] = []
    candidate_ids = set()
    for ctype in ["class", "function", "method", "variable", "segment"]:
        for cid in (candidate.get(ctype) or {}).keys():
            candidate_ids.add(cid)
    target_ids = (target_function_ids or set()) if target_function_ids is not None else set()

    for candidate_type in ["class", "function", "method", "variable", "segment"]:
        if candidate_type not in candidate:
            continue

        for component_id, component_info in candidate[candidate_type].items():
            if component_id in target_ids:
                continue
            component_name = component_id.split("@")[0]
            name_variants = (component_name, component_name.lower(), component_name.upper())
            should_match = False

            if any(v in text for text in component_names for v in name_variants):
                should_match = True

            if not should_match and candidate_type == "method":
                if "." in component_name:
                    method_name_only = component_name.split(".")[-1]
                    method_variants = (
                        method_name_only,
                        method_name_only.lower(),
                        method_name_only.upper(),
                    )
                    if any(v in text for text in component_names for v in method_variants):
                        should_match = True

            if should_match:
                if candidate_type == "method":
                    class_id = _method_to_class_id(component_id)
                    if class_id and class_id in candidate_ids:
                        class_info = (candidate.get("class") or {}).get(class_id)
                        if class_info and class_id not in target_ids:
                            matched_components.append({
                                "component_id": class_id,
                                "component_type": "class",
                                "source_code": class_info.get("source_code"),
                                "prompt": class_info.get("prompt"),
                                "score": float("inf"),
                                "references": class_info.get("references"),
                                "relative_path": class_info.get("relative_path"),
                                "is_third_party": class_info.get("is_third_party"),
                                "library": class_info.get("library"),
                            })
                    continue
                matched_components.append({
                    "component_id": component_id,
                    "component_type": candidate_type,
                    "source_code": component_info.get("source_code"),
                    "prompt": component_info.get("prompt"),
                    "score": float("inf"),
                    "references": component_info.get("references"),
                    "relative_path": component_info.get("relative_path"),
                    "is_third_party": component_info.get("is_third_party"),
                    "library": component_info.get("library"),
                })

                references = component_info.get("references")
                if isinstance(references, list) and len(references) > 0:
                    ref_dict = references[0]
                    if isinstance(ref_dict, str):
                        ref_id = ref_dict
                        ref_dict = {}
                    else:
                        ref_id = ref_dict.get("id") if isinstance(ref_dict, dict) else None

                    if ref_id and ref_id in candidate_ids and ref_id not in target_ids:
                        ref_type = ref_dict.get("component_type")

                        if ref_type in ["function", "method"]:
                            if ref_type == "method":
                                class_id = _method_to_class_id(ref_id)
                                if class_id and class_id in candidate_ids:
                                    class_info = (candidate.get("class") or {}).get(class_id)
                                    if class_info and class_id not in target_ids:
                                        ref_relative_path = class_id.split("@")[1] if "@" in class_id else ""
                                        extended_refs.append({
                                            "component_id": class_id,
                                            "component_type": "class",
                                            "source_code": class_info.get("source_code"),
                                            "prompt": class_info.get("source_code"),
                                            "score": ref_dict.get("score"),
                                            "relative_path": class_info.get("relative_path"),
                                            "is_third_party": class_info.get("is_third_party"),
                                            "library": class_info.get("library"),
                                        })
                                continue
                            ref_relative_path = ""
                            if "@" in ref_id:
                                ref_relative_path = ref_id.split("@")[1]
                            extended_refs.append({
                                "component_id": ref_id,
                                "component_type": ref_type,
                                "source_code": ref_dict.get("source_code"),
                                "prompt": ref_dict.get("source_code"),
                                "score": ref_dict.get("score"),
                                "relative_path": ref_dict.get("relative_path"),
                                "is_third_party": ref_dict.get("is_third_party"),
                                "library": ref_dict.get("library"),
                            })

    matched_components = [c for c in matched_components if c["component_id"] not in target_ids]
    extended_refs = [r for r in extended_refs if r["component_id"] not in target_ids]
    seen_m = set()
    seen_e = set()
    matched_components = [
        c for c in matched_components
        if c["component_id"] not in seen_m and not seen_m.add(c["component_id"])
    ]
    extended_refs = [
        r for r in extended_refs
        if r["component_id"] not in seen_e and not seen_e.add(r["component_id"])
    ]
    return matched_components, extended_refs


def extract_target_function_id(example: Dict[str, Any]) -> Optional[Set[str]]:
    """Extract target function component_id from example and return all possible formats."""
    target_function_prompt = example.get("target_function_prompt")
    relative_path = example.get("relative_path")
    component_type = example.get("type")
    repo_name = example.get("repo_name")

    if not target_function_prompt or not relative_path:
        return None

    if component_type == "function":
        prompt = target_function_prompt
    else:
        prompt = example.get("target_method_prompt")

    if not prompt:
        return None

    lines = prompt.strip().split("\n")
    func_name = None
    for line in lines:
        line = line.strip()
        if line.startswith("def "):
            func_name = line.split("def ")[1].split("(")[0].strip()
            break

    if not func_name:
        return None

    possible_ids = set()
    possible_ids.add(f"{func_name}@{relative_path}")

    if repo_name:
        possible_ids.add(f"{func_name}@{repo_name}/{relative_path}")
        possible_ids.add(f"{func_name}@{repo_name}/src/{relative_path}")
        if not relative_path.startswith("src/"):
            possible_ids.add(f"{func_name}@{repo_name}/src/{relative_path}")
        if relative_path.startswith("src/"):
            possible_ids.add(f"{func_name}@{repo_name}/{relative_path[4:]}")

    if not relative_path.startswith("src/"):
        possible_ids.add(f"{func_name}@src/{relative_path}")

    return possible_ids
