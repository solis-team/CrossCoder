import ast
import json
import os
import argparse
from typing import List, Set, Tuple

def is_primitive_assignment(node):
    if not isinstance(node, ast.Assign):
        return False
    
    if not node.value:
        return False
    
    primitive_types = (ast.List, ast.Set, ast.Dict, ast.Tuple, ast.Constant)
    
    if isinstance(node.value, primitive_types):
        return True
    
    if isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name):
            if node.value.func.id in ['list', 'set', 'dict', 'tuple', 'str', 'int', 'float', 'bool', 'bytes', 'bytearray', 'frozenset']:
                return True
    
    return False

def extract_calls_from_ast(node) -> Set[str]:
    calls = set()
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            calls.add(child.id)
        elif isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name):
                calls.add(child.value.id)
    
    return calls

def segment_code_with_calls(code: str) -> List[Tuple[str, Set[str]]]:
    tree = ast.parse(code)
    
    if not tree.body:
        return []
    
    if isinstance(tree.body[0], ast.FunctionDef):
        func_node = tree.body[0]
        statements = func_node.body
    else:
        statements = tree.body
    
    filtered_statements = []
    for stmt in statements:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str):
                continue
        filtered_statements.append(stmt)
    
    chunks = []
    primitive_buffer = []
    
    for stmt in filtered_statements:
        if is_primitive_assignment(stmt):
            primitive_buffer.append(stmt)
        else:
            if primitive_buffer:
                combined_nodes = primitive_buffer + [stmt]
                primitive_buffer = []
            else:
                combined_nodes = [stmt]
            
            start_line = combined_nodes[0].lineno
            end_line = combined_nodes[-1].end_lineno
            
            lines = code.split('\n')
            chunk_lines = lines[start_line - 1:end_line]
            chunk = '\n'.join(chunk_lines)
            
            calls = set()
            for node in combined_nodes:
                calls.update(extract_calls_from_ast(node))
            
            chunks.append((chunk, calls))
    
    if primitive_buffer:
        start_line = primitive_buffer[0].lineno
        end_line = primitive_buffer[-1].end_lineno
        lines = code.split('\n')
        chunk_lines = lines[start_line - 1:end_line]
        chunk = '\n'.join(chunk_lines)
        
        calls = set()
        for node in primitive_buffer:
            calls.update(extract_calls_from_ast(node))
        
        chunks.append((chunk, calls))
    
    return chunks

def is_third_party(relative_path: str, repo_name: str) -> bool:
    if not relative_path or not repo_name:
        return False
    
    first_folder = relative_path.split('/')[0] if '/' in relative_path else relative_path
    return first_folder != repo_name

def process_dependency_graph(graph_path: str, repo_name: str):
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    new_components = {}
    lg_counters = {}
    segment_mapping = {}
    
    for component_id, component in graph.items():
        relative_path = component.get('relative_path', '')
        component_type = component.get('component_type', '')
        
        should_segment = False
        if component_type == 'function' and relative_path and repo_name:
            first_folder = relative_path.split('/')[0] if '/' in relative_path else relative_path
            if first_folder == repo_name:
                should_segment = True
        
        if not should_segment:
            continue
        
        source_code = component.get('source_code', '')
        if not source_code:
            continue
        
        try:
            segments = segment_code_with_calls(source_code)
        except:
            continue
        
        original_outgoing = component.get('outgoing_calls', {})
        
        if relative_path not in lg_counters:
            lg_counters[relative_path] = 0
        
        component_segments = []
        
        for segment_code, call_pool in segments:
            lg_counters[relative_path] += 1
            segment_id = f"lg{lg_counters[relative_path]}@{relative_path}"
            component_segments.append(segment_id)
            
            segment_outgoing = {
                "class": [],
                "function": [],
                "method": [],
                "variable": []
            }
            
            for call_name in call_pool:
                for original_call in original_outgoing.get('class', []):
                    if original_call.split('@')[0] == call_name:
                        segment_outgoing['class'].append(original_call)
                        break
                
                for original_call in original_outgoing.get('function', []):
                    if original_call.split('@')[0] == call_name:
                        segment_outgoing['function'].append(original_call)
                        break
                
                for original_call in original_outgoing.get('method', []):
                    if original_call.split('@')[0] == call_name:
                        segment_outgoing['method'].append(original_call)
                        break
                
                for original_call in original_outgoing.get('variable', []):
                    if original_call.split('@')[0] == call_name:
                        segment_outgoing['variable'].append(original_call)
                        break
            
            new_components[segment_id] = {
                "id": segment_id,
                "component_type": "logic_segment",
                "file_path": component.get('file_path', ''),
                "relative_path": relative_path,
                "parent_component": component_id,
                "source_code": segment_code,
                "outgoing_calls": segment_outgoing
            }
            
            for call_type in ['class', 'function', 'method', 'variable']:
                for dep_id in segment_outgoing.get(call_type, []):
                    if dep_id not in segment_mapping:
                        segment_mapping[dep_id] = []
                    segment_mapping[dep_id].append(segment_id)
        
        noise = component.get('noise', {})
        for call_type in ['class', 'function', 'method', 'variable']:
            for dep_id in noise.get(call_type, []):
                if dep_id not in segment_mapping:
                    segment_mapping[dep_id] = []
                segment_mapping[dep_id].extend(component_segments)
    
    for component_id in graph:
        if component_id in segment_mapping:
            graph[component_id]['candidate_segment'] = list(set(segment_mapping[component_id]))
        else:
            graph[component_id]['candidate_segment'] = []
    
    graph.update(new_components)
    
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {graph_path}: added {len(new_components)} logic segments")
    print(f"Updated {len(segment_mapping)} components with candidate_segment field")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True, choices=['RepoExec', 'DevEval'])
    args = parser.parse_args()
    
    base_dir = f'data/parser_output/{args.benchmark}'
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found")
        return
    
    for repo_name in os.listdir(base_dir):
        repo_path = os.path.join(base_dir, repo_name)
        
        if not os.path.isdir(repo_path):
            continue
        
        graph_path = os.path.join(repo_path, 'dependency_graph.json')
        
        if not os.path.exists(graph_path):
            continue
        
        process_dependency_graph(graph_path, repo_name)

if __name__ == '__main__':
    main()



