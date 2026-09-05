# ----- Plan generation -----
PLAN_SYSTEM_PROMPT = """<ROLE>
You are an implementation-planning assistant. Your goal is to produce a **fine-grained, actionable implementation plan** for a target function body, given repository context (external libraries, repo structure, and repo signatures).
</ROLE>

<OBJECTIVE>
Produce an implementation plan that a developer can follow to implement the target function. You will receive:
- <THIRD_PARTY_LIBRARY>: third-party components by library — you may reference from here for your implementation plan
- <REPO_STRUCTURE>: structure tree of imported files and current file; captures relations (contains, invokes, inherits) and represents the dependency graph
- <REPO_SKETCH>: signatures of repo components (imported files and current file) — you may reference from here for your implementation plan
- <TARGET_FUNCTION_PROMPT>: the function to implement
</OBJECTIVE>

<INSTRUCTIONS>
Analyze the target function requirements and the provided context (THIRD_PARTY_LIBRARY, REPO_STRUCTURE, REPO_SKETCH). Generate a **fine-grained, detailed implementation plan** as a list of actionable steps for implementing the **function body code** (not the docstring).

**REQUIREMENTS:**
1. **Each step is about 5 lines of description** — each step will be used in the component-retrieval stage that follows, so write clearly and in enough detail for retrieval.
2. **Each step MUST describe specific components, functions, or operations** - NO generic descriptions like "implement core logic" or "process data"
3. **Each step should provide detailed explanation of what needs to be done**
4. **Be as fine-grained as possible** - break down complex operations into specific substeps
5. **Specify concrete actions** - mention exact variables, data structures, function calls, algorithms, or techniques to use
6. **Each step should be actionable** - a developer should know exactly what code to write from your description
7. **Focus on the implementation code** - do not generate steps for writing docstrings or comments
8. **You should consider edge cases and error handling** - include steps for validating inputs, handling exceptions, and ensuring robustness
9. **You should explicitly mention potential dependencies** — predefined functions, classes, or variables from the provided sketches that the implementation may use; reference them using the format `name` (wrap the exact identifier name in backticks, e.g., `function_name`, `ClassName`, `variable_name`).
10. **You should explicitly mention potential similar code** in the repo (e.g. functions that do related things, same patterns) when they could help generation; reference them using the format `name` (backticks) just like dependencies.

**Output Format:**
Wrap your implementation plan in <OUTPUT> and </OUTPUT> tags. Inside the tags, provide a Python list of strings where each string is one detailed implementation step.

<OUTPUT>
```python
["Step 1 description...", "Step 2 description...", "Step 3 description...", ...]
```
</OUTPUT>
</INSTRUCTIONS>"""

PLAN_USER_PROMPT = """# Task: Generate Detailed Implementation Plan

<THIRD_PARTY_LIBRARY>

{external_information}

</THIRD_PARTY_LIBRARY>
{repo_structure_block}{repo_sketch_block}
<TARGET_FUNCTION_PROMPT>
{target_prompt}
</TARGET_FUNCTION_PROMPT>

Generate the implementation plan following the instructions."""


PLAN_USER_PROMPT_NO_CONTEXT = """# Task: Generate Detailed Implementation Plan

## Target Function to Implement

{target_prompt}

Generate the implementation plan following the instructions."""


def get_plan_user_prompt(
    external_information: str,
    repo_sketch: str,
    repo_structure: str,
    target_prompt: str,
    have_candidates: bool,
) -> str:
    """Build the full user prompt for plan generation by formatting PLAN_USER_PROMPT or PLAN_USER_PROMPT_NO_CONTEXT."""
    if have_candidates:
        repo_structure_block = (
            f"\n<REPO_STRUCTURE>\n{repo_structure}\n\n</REPO_STRUCTURE>\n"
            if repo_structure
            else ""
        )
        repo_sketch_block = (
            f"\n<REPO_SKETCH>\n\n{repo_sketch}\n\n</REPO_SKETCH>\n"
            if repo_sketch
            else ""
        )
        return PLAN_USER_PROMPT.format(
            external_information=external_information,
            repo_structure_block=repo_structure_block,
            repo_sketch_block=repo_sketch_block,
            target_prompt=target_prompt,
        )
    return PLAN_USER_PROMPT_NO_CONTEXT.format(target_prompt=target_prompt)


def _score_for_compare(score) -> float:
    """Normalize score for comparison/sort: 'inf' (str) or float('inf') -> inf, else float(score)."""
    if score == "inf" or score == float("inf"):
        return float("inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("inf")

CODEGEN_SYSTEM_PROMPT = """<ROLE>
You are a helpful coding assistant.
</ROLE>

<OBJECTIVE>
Reason about the task and write Python code to complete the target function given in the user message. You may use the context provided in the user message to complete the task.
</OBJECTIVE>

<OUTPUT>
You must include the full function: signature, docstring, and function body. Put your solution inside a ```python code block.
</OUTPUT>"""


def build_final_user_prompt_from_plan(sample: dict, include_tree: bool = True) -> str:
    intro = "You are a Python programmer. Here is all the context you may find useful to complete the function:"
    pred = sample.get("predicted_components") or []
    current_rel_path = sample.get("relative_path") or ""

    lib_class_groups: dict = {}
    lib_fv_groups: dict = {}
    file_groups: dict = {}
    current_file_components: list = []

    for c in pred:
        score = _score_for_compare(c.get("score"))
        source_code = c.get("source_code") or ""
        if c.get("is_third_party"):
            lib_name = c.get("lib_name") or c.get("library") or "external"
            if c.get("type") == "class":
                lib_class_groups.setdefault(lib_name, []).append((score, source_code))
            else:
                lib_fv_groups.setdefault(lib_name, []).append((score, source_code))
        else:
            rel_path = c.get("relative_path") or ""
            if rel_path == current_rel_path:
                current_file_components.append((score, source_code))
            else:
                file_groups.setdefault(rel_path, []).append((score, source_code))

    def avg_score(entries):
        scores = [s for s, _ in entries if s != float("inf")]
        if not scores:
            return float("inf")
        return sum(scores) / len(scores)

    parts = [intro, ""]

    sorted_lib_class = sorted(lib_class_groups.keys(), key=lambda n: avg_score(lib_class_groups[n]))
    for lib_name in sorted_lib_class:
        parts.append(f"#LIB {lib_name}")
        for _, source_code in lib_class_groups[lib_name]:
            parts.append(source_code)
            parts.append("")

    ranked_groups = []
    for rel_path, entries in file_groups.items():
        ranked_groups.append(("file", rel_path, entries))
    for lib_name, entries in lib_fv_groups.items():
        ranked_groups.append(("lib_fv", lib_name, entries))

    ranked_groups.sort(key=lambda x: avg_score(x[2]))

    for kind, key, entries in ranked_groups:
        if kind == "file":
            parts.append(f"#FILE {key}")
        else:
            parts.append(f"#LIB {key}")
        for _, source_code in entries:
            parts.append(source_code)
            parts.append("")

    import_stmts = sample.get("import_statements") or []
    if current_file_components or import_stmts:
        parts.append(f"#CURRENT_FILE {current_rel_path}")
        if import_stmts:
            parts.append("\n".join(import_stmts))
            parts.append("")
        for _, source_code in current_file_components:
            parts.append(source_code)
            parts.append("")

    clean_name = sample.get("clean_name") or sample.get("full_name", "").split("@")[0]
    parts.append(
        f"Based on the information above, please complete the function {clean_name} in {current_rel_path}:"
    )

    if sample.get("type") == "method":
        target_prompt = sample.get("target_method_prompt")
    else:
        target_prompt = sample.get("target_function_prompt")

    parts.append("")
    parts.append(target_prompt)
    return "\n".join(parts)


def _type_rank(comp_type: str) -> int:
    """Sort key for type tiebreaking: class < variable < function (lower = earlier in prompt)."""
    return {"class": 0, "variable": 1, "function": 2, "method": 2, "segment": 2}.get(comp_type, 3)


def build_reranked_user_prompt_versionexec(sample: dict, max_tokens: int = 4096, threshold: float = 0.25) -> str:
    from utils.token import count_tokens

    intro = "You are a Python programmer. Here is all the context you may find useful to complete the function:"
    pred = sample.get("predicted_components") or []
    target_prompt = sample.get("target_function_prompt") or ""

    part_target = ["Based on the information above, please complete the function:", "", target_prompt]

    candidates = []
    for c in pred:
        comp_name = (c.get("id") or "").split("@")[0]
        if comp_name.startswith("__all__"):
            continue
        score = _score_for_compare(c.get("score"))
        if score < threshold:
            continue
        candidates.append((score, _type_rank(c.get("type") or ""), c))
    candidates.sort(key=lambda x: (x[0], x[1]))  # ascending score, stable on dict tiebreak

    selected = []

    def _total():
        lines = [intro, ""]
        for node in selected:
            lines += node
        return lines + part_target

    for score, _, c in reversed(candidates):
        lib_name = c.get("lib_name") or c.get("library") or "external"
        node = [f"#LIB {lib_name}", c.get("source_code") or "", ""]
        if count_tokens("\n".join(_total() + node)) > max_tokens:
            continue
        selected.append(node)

    selected.reverse()

    parts = [intro, ""]
    for node in selected:
        parts += node
    parts += part_target
    return "\n".join(parts)


def build_reranked_user_prompt_from_plan(sample: dict, max_tokens: int = 4096, include_import: bool = True, threshold: float = 0.25, is_versionexec: bool = False) -> str:
    if is_versionexec:
        return build_reranked_user_prompt_versionexec(sample, max_tokens=max_tokens, threshold=threshold)

    from utils.token import count_tokens

    intro = "You are a Python programmer working with a repository. Here is all the context you may find useful to complete the function:"
    pred = sample.get("predicted_components") or []
    current_rel_path = sample.get("relative_path") or ""
    import_stmts = sample.get("import_statements") or [] if include_import else []
    current_file_gk = ("file", current_rel_path)

    third_party_classes = []
    base_components = []       
    remaining_components = []  

    for c in pred:
        comp_name = (c.get("id") or "").split("@")[0]
        if comp_name.startswith("__all__"):
            continue
        score = _score_for_compare(c.get("score"))
        if score < threshold:
            continue
        is_third_party = c.get("is_third_party", False)
        comp_type = c.get("type") or ""
        if is_third_party and comp_type == "class":
            third_party_classes.append((score, c))
        elif is_third_party and comp_type in ("function", "variable") and score != 1.0:
            third_party_classes.append((score, c))
        elif score >= 1.0:
            base_components.append((score, c))
        else:
            remaining_components.append((score, c))

    def sort_key(item):
        score, c = item
        s = score if score != float("inf") else 1e18
        return (s, _type_rank(c.get("type") or ""))

    third_party_classes.sort(key=sort_key)
    base_components.sort(key=sort_key)
    remaining_components.sort(key=sort_key)

    base_groups = {}
    base_group_order = []
    for item in base_components:
        sk = sort_key(item)
        _, c = item
        gk = ("lib", c.get("lib_name") or c.get("library") or "external") if c.get("is_third_party") \
             else ("file", c.get("relative_path") or "")
        if gk not in base_groups:
            base_groups[gk] = []
            base_group_order.append(gk)
        base_groups[gk].append((sk, c))

    for gk in base_groups:
        base_groups[gk].sort(key=lambda x: x[0])
    base_group_order.sort(key=lambda gk: base_groups[gk][0][0])

    base_group_order_display = [gk for gk in base_group_order if gk != current_file_gk]
    if current_file_gk in base_groups:
        base_group_order_display.append(current_file_gk)

    import_in_base = bool(import_stmts) and current_file_gk in base_groups
    import_in_remaining = bool(import_stmts) and not import_in_base

    def _build_base_group(gk):
        kind, name = gk
        header = f"#LIB {name}" if kind == "lib" else (
            f"#CURRENT FILE {name}" if name == current_rel_path else f"#FILE {name}"
        )
        lines = [header]
        if gk == current_file_gk and import_in_base:
            lines.append("\n".join(import_stmts))
            lines.append("")
        for _, c in base_groups[gk]:
            lines.append(c.get("source_code") or "")
            lines.append("")
        return lines

    part2b_lines = []
    for gk in base_group_order_display:
        part2b_lines += _build_base_group(gk)

    clean_name = sample.get("clean_name") or sample.get("full_name", "").split("@")[0]
    if sample.get("type") == "method":
        target_prompt = sample.get("target_method_prompt")
    else:
        target_prompt = sample.get("target_function_prompt")

    part3_lines = [
        f"Based on the information above, please complete the function {clean_name} in the current file {current_rel_path}:",
        "",
        target_prompt,
        "",
        "Note: Do not write any import statements. All necessary imports are already available in the codebase. Just write the target function.",
    ]

    fixed_suffix = part2b_lines + part3_lines

    sel_remaining = []  # list of (sk, node_lines)

    def _remaining_total():
        lines = [intro, ""]
        for _, node in sel_remaining:
            lines += node
        return lines + fixed_suffix

    for item in reversed(remaining_components):
        score, c = item
        sk = sort_key(item)
        source_code = c.get("source_code") or ""
        if c.get("is_third_party"):
            lib_name = c.get("lib_name") or c.get("library") or "external"
            node = [f"#LIB {lib_name}", source_code, ""]
        else:
            rel_path = c.get("relative_path") or ""
            header = f"#CURRENT FILE {rel_path}" if rel_path == current_rel_path else f"#FILE {rel_path}"
            node = [header, source_code, ""]
        if count_tokens("\n".join(_remaining_total() + node)) > max_tokens:
            continue
        sel_remaining.append((sk, node))

    sel_remaining.sort(key=lambda x: x[0])

    # Inject import_stmts into first current_file node in Part 2a (if needed)
    if import_in_remaining:
        for i, (sk, node) in enumerate(sel_remaining):
            if node and node[0] == f"#CURRENT FILE {current_rel_path}":
                sel_remaining[i] = (sk, [node[0], "\n".join(import_stmts), ""] + node[1:])
                break

    sel_tp = []  # list of (sk, node_lines)

    def _tp_total():
        lines = [intro, ""]
        for _, node in sel_tp:
            lines += node
        for _, node in sel_remaining:
            lines += node
        return lines + fixed_suffix

    for item in reversed(third_party_classes):
        score, c = item
        sk = sort_key(item)
        lib_name = c.get("lib_name") or c.get("library") or "external"
        node = [f"#LIB {lib_name}", c.get("source_code") or "", ""]
        if count_tokens("\n".join(_tp_total() + node)) > max_tokens:
            continue
        sel_tp.append((sk, node))

    sel_tp.sort(key=lambda x: x[0])

    parts = [intro, ""]
    for _, node in sel_tp:
        parts += node
    for _, node in sel_remaining:
        parts += node
    parts += fixed_suffix

    return "\n".join(parts)
