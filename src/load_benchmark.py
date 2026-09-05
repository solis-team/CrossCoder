import os
import sys
import json
import glob
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import datasets
from abc import ABC, abstractmethod
import ast

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from utils.build_tree_structure import build_tree_structure_for_sample


class Benchmark_Loader(ABC):
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None, model: str = None, candidate_only: bool = False):
        self.graphs_base_dir = graphs_base_dir
        self.output_dir = output_dir or "data"
        self.dependency_cache = {}
        self.external_knowledge_cache = {}
        self.model = model
        self.candidate_only = candidate_only
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_dependency_graph(self, graph_path: str) -> Dict[str, Any]:
        """Load dependency graph from JSON file with caching."""
        if graph_path in self.dependency_cache:
            return self.dependency_cache[graph_path]
        
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            self.dependency_cache[graph_path] = graph_data
            return graph_data
        except Exception as e:
            print(f"Error loading dependency graph from {graph_path}: {e}")
            return {}
    
    def load_external_knowledge(self, repo_dir: str) -> Dict[str, Any]:
        """Load external knowledge from JSON file in repo directory with caching."""
        external_path = os.path.join(repo_dir, "external_knowledge.json")
        
        if external_path in self.external_knowledge_cache:
            return self.external_knowledge_cache[external_path]
        
        try:
            with open(external_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            self.external_knowledge_cache[external_path] = knowledge_data
            return knowledge_data
        except Exception as e:
            print(f"Error loading external knowledge from {external_path}: {e}")
            return {}
    
    def get_import_statements(self, external_knowledge: Dict[str, Any], relative_path: str) -> List[str]:
        """Get import statements for a given relative path."""
        file_data = external_knowledge.get(relative_path, {})
        return file_data.get('import_statements', [])
    
    def find_dependency_graph_path(self, repo_name: str) -> Optional[str]:
        """Find dependency graph file for a given repository name."""
        pattern = os.path.join(self.graphs_base_dir, "**", repo_name, "dependency_graph.json")
        matches = glob.glob(pattern, recursive=True)
        
        if matches:
            return matches[0]
        
        pattern = os.path.join(self.graphs_base_dir, "**", "dependency_graph.json")
        all_graphs = glob.glob(pattern, recursive=True)
        
        for graph_path in all_graphs:
            if repo_name in graph_path:
                return graph_path
        
        return None
    
    def format_target_component(self, comp_data: Dict[str, Any]) -> str:
        """Format target component with name and description only."""
        result = {
            'name': comp_data['id'],
            'description': comp_data.get('docstring', 'DOCSTRING')
        }
        
        signature = comp_data.get('signature', '')
        if signature:
            result['signature'] = signature
        
        return json.dumps(result, ensure_ascii=False)
    
    def format_function_class_component(self, comp_data: Dict[str, Any]) -> str:
        """Format function or class component as JSON text."""
        result = {
            'name': comp_data['id'],
            'description': comp_data.get('docstring', 'DOCSTRING'),
            'code': comp_data.get('source_code', '')
        }
        
        if comp_data['component_type'] in ['function', 'class']:
            signature = comp_data.get('signature', '')
            if signature:
                result['signature'] = signature
        
        return json.dumps(result, ensure_ascii=False)
    
    def create_class_sketch(self, class_source_code: str, target_method_name: str) -> str:
        try:
            tree = ast.parse(class_source_code)
            
            if not tree.body or not isinstance(tree.body[0], ast.ClassDef):
                return class_source_code
            
            class_node = tree.body[0]
            lines = class_source_code.split('\n')
            
            sketch_parts = []
            
            class_start = class_node.lineno - 1
            class_def_line = lines[class_start]
            sketch_parts.append(class_def_line)
            
            if class_node.body and isinstance(class_node.body[0], ast.Expr):
                if isinstance(class_node.body[0].value, ast.Constant):
                    if isinstance(class_node.body[0].value.value, str):
                        docstring_start = class_node.body[0].lineno - 1
                        docstring_end = class_node.body[0].end_lineno
                        sketch_parts.extend(lines[docstring_start:docstring_end])
            
            for item in class_node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    method_start = item.lineno - 1
                    method_end = item.end_lineno
                    
                    if method_name == '__init__' or method_name == target_method_name:
                        sketch_parts.extend(lines[method_start:method_end])
                    else:
                        signature_line = lines[method_start]
                        sketch_parts.append(signature_line)
                        
                        if item.body and isinstance(item.body[0], ast.Expr):
                            if isinstance(item.body[0].value, ast.Constant):
                                if isinstance(item.body[0].value.value, str):
                                    docstring_start = item.body[0].lineno - 1
                                    docstring_end = item.body[0].end_lineno
                                    sketch_parts.extend(lines[docstring_start:docstring_end])
                        
                        indent = len(signature_line) - len(signature_line.lstrip())
                        sketch_parts.append(' ' * (indent + 4) + '...')
            
            return '\n'.join(sketch_parts)
        
        except:
            return class_source_code
    
    def format_context(self, candidate_comp: Dict[str, Any], components: Dict[str, Any], comp_type: str) -> str:
        if comp_type == 'segment':
            parent_component_id = candidate_comp.get('parent_component', '')
            if parent_component_id and parent_component_id in components:
                parent_comp = components[parent_component_id]
                return parent_comp.get('source_code', '')
            return candidate_comp.get('source_code', '')
        
        elif comp_type == 'method':
            method_id = candidate_comp.get('id', '')
            method_source = candidate_comp.get('source_code', '')
            
            if '@' in method_id:
                parts = method_id.split('@')
                if len(parts) >= 1 and '.' in parts[0]:
                    method_parts = parts[0].split('.')
                    if len(method_parts) >= 2:
                        class_name = method_parts[0]
                        method_name = method_parts[1]
                        relative_path = '@'.join(parts[1:]) if len(parts) > 1 else ''
                        
                        if relative_path:
                            class_id = f"{class_name}@{relative_path}"
                            if class_id in components:
                                class_comp = components[class_id]
                                class_source = class_comp.get('source_code', '')
                                return self.create_class_sketch(class_source, method_name)
            
            return method_source
        
        return candidate_comp.get('source_code', '')
    
    def get_same_path_candidate_ids(
        self,
        components: Dict[str, Any],
        target_relative_path: str,
        target_component_id: str,
    ) -> Dict[str, List[str]]:
        """Return comp_type -> list of component ids in graph with same relative_path and id != target_component_id."""
        result = {'class': [], 'function': [], 'method': [], 'variable': [], 'segment': []}
        for comp_id, comp in components.items():
            if comp_id == target_component_id:
                continue
            if comp.get('relative_path') != target_relative_path:
                continue
            ctype = comp.get('component_type')
            if ctype in result:
                result[ctype].append(comp_id)
        return result

    def get_external_entity_candidate_ids(
        self,
        external_knowledge: Dict[str, Any],
        target_relative_path: str,
        target_component_id: str,
        components: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Classify external_entity (imported into this file) into class/function/variable; exclude target_component_id.
        Only includes entities that exist in dependency_graph (components)."""
        result = {'class': [], 'function': [], 'variable': []}
        file_data = external_knowledge.get(target_relative_path, {})
        entity_ids = file_data.get('external_entity', [])
        for eid in entity_ids:
            if eid == target_component_id:
                continue
            if eid not in components:
                continue
            ctype = components[eid].get('component_type')
            if ctype == 'class':
                result['class'].append(eid)
            elif ctype in ('function', 'method'):
                result['function'].append(eid)
            elif ctype == 'variable':
                result['variable'].append(eid)
        return result

    def get_method_candidates_from_classes(self, class_candidates: List[str], components: Dict[str, Any]) -> List[str]:
        """Get method candidates from class candidates.
        
        For each class candidate like 'MyClass@file.py', find methods in components
        that match pattern 'MyClass.method@file.py'.
        """
        method_candidates = []
        
        for class_id in class_candidates:
            if '@' not in class_id:
                continue
            
            class_name, file_path = class_id.split('@', 1)
            
            for comp_id, comp in components.items():
                if comp.get('component_type') != 'method':
                    continue
                
                if comp_id.startswith(f"{class_name}.") and comp_id.endswith(f"@{file_path}"):
                    method_candidates.append(comp_id)
        
        return method_candidates
    
    def process_candidates(self, target_comp: Dict[str, Any], components: Dict[str, Any], 
                          candidate_ids: List[str], comp_type: str, repo_dir: str = None) -> Dict[str, Dict[str, Any]]:
        candidates = {}
        
        current_repo_name = None
        if repo_dir:
            current_repo_name = os.path.basename(repo_dir)
        
        target_id = target_comp.get('id', '')
        
        for candidate_id in candidate_ids:
            if candidate_id in components:
                candidate_comp = components[candidate_id]
                relative_path = candidate_comp.get('relative_path', '')
                
                is_third_party = False
                library = None
                
                if relative_path and current_repo_name:
                    first_folder = relative_path.split('/')[0] if '/' in relative_path else relative_path
                    if first_folder != current_repo_name:
                        is_third_party = True
                        library = first_folder

                if is_third_party:
                    if (
                        candidate_id.startswith('_') or
                        '@_' in candidate_id or
                        '/_' in candidate_id
                    ):
                        continue
                
                references = []
                for comp_id, comp in components.items():
                    comp_type_check = comp.get('component_type')
                    if comp_type_check in ['function', 'method'] and comp_id != target_id:
                        outgoing_calls = comp.get('outgoing_calls', {})
                        
                        found = False
                        for call_type in ['class', 'function', 'method', 'variable']:
                            if candidate_id in outgoing_calls.get(call_type, []):
                                found = True
                                break
                        
                        if found:
                            references.append(comp_id)
                
                docstring = candidate_comp.get('docstring', '')
                candidate_dict = {
                    'relative_path': relative_path,
                    'source_code': candidate_comp.get('source_code', ''),
                    'docstring': docstring,
                    'description': docstring,
                    'is_third_party': is_third_party,
                    'library': library,
                    'references': references
                }
                
                if comp_type in ['method', 'segment']:
                    prompt = self.format_context(candidate_comp, components, comp_type)
                    candidate_dict['prompt'] = prompt
                
                candidates[candidate_id] = candidate_dict
        
        return candidates

    def filter_references_to_candidates(self, candidate_dict: Dict[str, Dict[str, Any]]) -> None:
        """In-place: keep only reference ids that are in candidate_dict (current + imported)."""
        valid_ids = set()
        for ctype in ['class', 'function', 'method', 'variable', 'segment']:
            valid_ids.update((candidate_dict.get(ctype) or {}).keys())
        for ctype in ['class', 'function', 'method', 'variable', 'segment']:
            for cand_info in (candidate_dict.get(ctype) or {}).values():
                refs = cand_info.get('references', [])
                if not refs:
                    continue
                if isinstance(refs[0], dict):
                    cand_info['references'] = [r for r in refs if r.get('id') in valid_ids]
                else:
                    cand_info['references'] = [r for r in refs if r in valid_ids]

    @abstractmethod
    def load_dataset(self) -> Any:
        """Load the benchmark dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_output_filename(self) -> str:
        """Get the output filename for processed data. Must be implemented by subclasses."""
        pass
    
    def get_last_processed_id(self, output_path: str) -> Optional[str]:
        """Get the last processed task ID from existing output file."""
        if not os.path.exists(output_path):
            return None
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                last_valid_id = None
                line_num = 0
                
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        sample = json.loads(line)
                        sample_id = sample.get('id')
                        if sample_id is not None:
                            last_valid_id = sample_id
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}, stopping at last valid ID: {last_valid_id}")
                        break
                
                return last_valid_id
        except Exception as e:
            print(f"Error reading existing output file: {e}")
        
        return None
    
    def should_process_sample(self, sample_id: str, last_processed_id: Optional[str]) -> bool:
        """Check if sample should be processed based on last processed ID."""
        if last_processed_id is None:
            return True
        
        if isinstance(sample_id, str) and sample_id.isdigit():
            sample_id = int(sample_id)
        if isinstance(last_processed_id, str) and last_processed_id.isdigit():
            last_processed_id = int(last_processed_id)
        
        try:
            return sample_id > last_processed_id
        except:
            return str(sample_id) > str(last_processed_id)

    def process_dataset(self, max_samples: Optional[int] = None):
        """Process the entire dataset and save to JSONL format."""
        output_path = os.path.join(self.output_dir, self.get_output_filename())
        
        if self.candidate_only:
            if not os.path.exists(output_path):
                print(f"Error: Processed file not found: {output_path}")
                print("Please run without --candidate_only first to create the processed file")
                return
            
            print("=" * 60)
            print("Candidate-only mode: Reloading candidates from dependency graphs")
            print("=" * 60)
            
            processed_samples = []
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading existing processed samples"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        processed_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
            
            print(f"Loaded {len(processed_samples)} existing samples")
            print("Reloading candidates from dependency graphs...")
            
            processed_count = 0
            failed_count = 0
            total_candidates = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for existing_sample in tqdm(processed_samples, desc="Reloading candidates"):
                    try:
                        updated_sample = existing_sample.copy()
                        
                        target_comp = updated_sample.get('target_component', {})
                        if not target_comp:
                            json.dump(updated_sample, f, ensure_ascii=False)
                            f.write('\n')
                            continue
                        
                        repo_name = updated_sample.get('repo_name')
                        if not repo_name:
                            json.dump(updated_sample, f, ensure_ascii=False)
                            f.write('\n')
                            continue
                        
                        graph_path = self.find_dependency_graph_path(repo_name)
                        if not graph_path:
                            json.dump(updated_sample, f, ensure_ascii=False)
                            f.write('\n')
                            continue
                        
                        components = self.load_dependency_graph(graph_path)
                        repo_dir = os.path.dirname(graph_path)
                        
                        if target_comp.get('id') not in components:
                            json.dump(updated_sample, f, ensure_ascii=False)
                            f.write('\n')
                            continue
                        
                        target_comp_from_graph = components[target_comp.get('id')]
                        outgoing_calls = target_comp_from_graph.get('outgoing_calls', {"class": [], "function": [], "method": [], "variable": []})
                        noise = target_comp_from_graph.get('noise', {"class": [], "function": [], "method": [], "variable": []})
                        updated_sample['g_dependency'] = list(dict.fromkeys(
                            list(outgoing_calls.get('class', [])) +
                            list(outgoing_calls.get('function', [])) +
                            list(outgoing_calls.get('variable', []))
                        ))
                        
                        target_relative_path = target_comp_from_graph.get('relative_path', '')
                        same_path = self.get_same_path_candidate_ids(components, target_relative_path, target_comp.get('id'))
                        external_knowledge = self.load_external_knowledge(repo_dir)
                        external_entity = self.get_external_entity_candidate_ids(
                            external_knowledge, target_relative_path, target_comp.get('id'), components
                        )
                        all_candidates = {
                            'class': list(set(outgoing_calls.get('class', []) + noise.get('class', [])) | set(same_path.get('class', [])) | set(external_entity.get('class', []))),
                            'function': list(set(outgoing_calls.get('function', []) + noise.get('function', [])) | set(same_path.get('function', [])) | set(external_entity.get('function', []))),
                            'variable': list(set(outgoing_calls.get('variable', []) + noise.get('variable', [])) | set(same_path.get('variable', [])) | set(external_entity.get('variable', []))),
                            'segment': list(set(target_comp_from_graph.get('candidate_segment', [])) | set(same_path.get('segment', []))),
                        }
                        
                        new_candidates = {
                            'class': self.process_candidates(target_comp_from_graph, components, all_candidates['class'], 'class', repo_dir),
                            'function': self.process_candidates(target_comp_from_graph, components, all_candidates['function'], 'function', repo_dir),
                            'variable': self.process_candidates(target_comp_from_graph, components, all_candidates['variable'], 'variable', repo_dir),
                            'segment': self.process_candidates(target_comp_from_graph, components, all_candidates['segment'], 'segment', repo_dir)
                        }
                        
                        class_candidate_ids = list(new_candidates['class'].keys())
                        method_candidate_ids = self.get_method_candidates_from_classes(class_candidate_ids, components)
                        new_candidates['method'] = self.process_candidates(target_comp_from_graph, components, method_candidate_ids, 'method', repo_dir)
                        
                        if not self.extend:
                            for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
                                new_candidates[comp_type] = {
                                    cand_id: cand_info
                                    for cand_id, cand_info in new_candidates[comp_type].items()
                                    if not cand_info.get('is_third_party', False)
                                }

                        self.filter_references_to_candidates(new_candidates)
                        updated_sample['candidate'] = new_candidates

                        try:
                            updated_sample['tree_structure'] = build_tree_structure_for_sample(
                                updated_sample, components, invokes_max_depth=5
                            )
                        except Exception as e:
                            print(f"Warning: tree_structure build failed for sample {updated_sample.get('id')}: {e}")

                        json.dump(updated_sample, f, ensure_ascii=False)
                        f.write('\n')
                        f.flush()
                        processed_count += 1

                        if 'candidate' in updated_sample:
                            for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
                                if comp_type in updated_sample['candidate']:
                                    total_candidates += len(updated_sample['candidate'][comp_type])
                    except Exception as e:
                        print(f"Error processing sample {existing_sample.get('id', 'unknown')}: {e}")
                        failed_count += 1
                        json.dump(existing_sample, f, ensure_ascii=False)
                        f.write('\n')
                    
                    if processed_count % 50 == 0 and len(self.dependency_cache) > 10:
                        self.dependency_cache.clear()
                        self.external_knowledge_cache.clear()
            
            print(f"Successfully reloaded candidates for: {processed_count} samples")
            print(f"Failed to reload: {failed_count} samples")
            avg_candidates = total_candidates / processed_count if processed_count else 0
            print(f"Average candidates per sample: {avg_candidates:.2f}")
            print(f"Saved updated samples to {output_path}")
            return
        
        dataset = self.load_dataset()
        if dataset is None:
            print("Failed to load dataset")
            return
        
        if max_samples and hasattr(dataset, '__len__'):
            dataset = dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
            print(f"Processing first {len(dataset)} samples")
        
        last_processed_id = self.get_last_processed_id(output_path)
        
        if last_processed_id:
            print(f"Resuming from last processed ID: {last_processed_id}")
            file_mode = 'a'
        else:
            print("Starting fresh processing")
            file_mode = 'w'
        
        processed_count = 0
        failed_count = 0
        total_candidates = 0
        skipped_count = 0
        
        print("Processing samples...")
        if isinstance(dataset, list):
            iterator = tqdm(dataset)
        else:
            iterator = tqdm(dataset)
        
        with open(output_path, file_mode, encoding='utf-8') as f:
            for sample in iterator:
                sample_id = sample.get('id') if isinstance(sample, dict) else str(sample)
                
                if not self.should_process_sample(sample_id, last_processed_id):
                    skipped_count += 1
                    continue
                
                processed_sample = self.process_sample(sample)
                
                if processed_sample:
                    repo_name = processed_sample.get('repo_name')
                    if processed_sample.get('candidate') and repo_name:
                        graph_path = self.find_dependency_graph_path(repo_name)
                        if graph_path:
                            graph = self.load_dependency_graph(graph_path)
                            if graph:
                                try:
                                    processed_sample['tree_structure'] = build_tree_structure_for_sample(
                                        processed_sample, graph, invokes_max_depth=5
                                    )
                                except Exception as e:
                                    print(f"Warning: tree_structure build failed for sample {sample_id}: {e}")
                    json.dump(processed_sample, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    processed_count += 1
                    
                    if 'candidate' in processed_sample:
                        for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
                            if comp_type in processed_sample['candidate']:
                                total_candidates += len(processed_sample['candidate'][comp_type])
                else:
                    failed_count += 1
                
                if processed_count % 50 == 0 and len(self.dependency_cache) > 10:
                    self.dependency_cache.clear()
                    self.external_knowledge_cache.clear()
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed samples")
        print(f"Successfully processed: {processed_count} samples")
        print(f"Failed to process: {failed_count} samples")
        
        if processed_count == 0:
            print("No new samples were processed")
            return
        
        avg_candidates = total_candidates / processed_count if processed_count else 0
        print(f"Average candidates per sample: {avg_candidates:.2f}")
        print(f"Saved {processed_count} processed samples to {output_path}")
    
    def save_processed_benchmark(self, processed_samples: List[Dict[str, Any]]):
        """Save processed samples to JSONL file."""
        pass


class RepoExec_Loader(Benchmark_Loader):
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None,
                 dataset_name: str = "Fsoft-AIC/RepoExec", split: str = "full_context", extend: bool = False, model: str = None, candidate_only: bool = False):
        super().__init__(graphs_base_dir, output_dir, model, candidate_only)
        self.dataset_name = dataset_name
        self.split = split
        self.extend = extend
    
    def extract_repo_name(self, project: str) -> str:
        """Extract repository name from project field."""
        return project.split('/')[1]
    
    def construct_component_id(self, entry_point: str, module: str, repo_name: str) -> str:
        """Construct component ID from entry_point and module."""
        module_path = module.replace('.', '/')
        return [f"{entry_point}@{repo_name}/{module_path}.py", f"{entry_point}@{repo_name}/src/{module_path}.py"]
    
    def load_dataset(self) -> datasets.Dataset:
        """Load RepoExec dataset from HuggingFace."""
        try:
            print(f"Loading dataset {self.dataset_name}, split {self.split}...")
            dataset = datasets.load_dataset(self.dataset_name, split=self.split)
            print(f"Loaded {len(dataset)} samples from RepoExec")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single RepoExec sample and create the required format."""
        try:
            task_id = sample['id']
            target_function_prompt = sample.get('target_function_prompt')
            project = sample['project']
            module = sample['module']
            entry_point = sample['entry_point']
            implementation_plan = sample.get('implementation_plan', [])
            
            repo_name = self.extract_repo_name(project)
            #print(repo_name)
            graph_path = self.find_dependency_graph_path(repo_name)
            
            if not graph_path:
                print(f"Warning: No dependency graph found for repo {repo_name}")
                # print(project)
                return None

            # print(graph_path)

            components = self.load_dependency_graph(graph_path)
            repo_dir = os.path.dirname(graph_path)
            external_knowledge = self.load_external_knowledge(repo_dir)
            
            target_component_ids = self.construct_component_id(entry_point, module, repo_name)
            target_component_id = target_component_ids[0]

            if target_component_id not in components:
                module_path = module.replace('.', '/') + '.py'
                if target_component_ids[1] in components:
                    target_component_id = target_component_ids[1]
                elif module_path in components:
                    target_component_id = module_path
                else:
                    print(f"Warning: Target component {target_component_id} not found in dependency graph")
                    return None
            
            target_comp = components[target_component_id]
            target_relative_path = target_comp.get('relative_path', '')
            
            outgoing_calls = target_comp.get('outgoing_calls', {"class": [], "function": [], "method": [], "variable": []})
            noise = target_comp.get('noise', {"class": [], "function": [], "method": [], "variable": []})
            g_dependency = list(dict.fromkeys(
                list(outgoing_calls.get('class', [])) +
                list(outgoing_calls.get('function', [])) +
                list(outgoing_calls.get('variable', []))
            ))
            
            same_path = self.get_same_path_candidate_ids(components, target_relative_path, target_component_id)
            external_entity = self.get_external_entity_candidate_ids(
                external_knowledge, target_relative_path, target_component_id, components
            )
            all_candidates = {
                'class': list(set(outgoing_calls.get('class', []) + noise.get('class', [])) | set(same_path.get('class', [])) | set(external_entity.get('class', []))),
                'function': list(set(outgoing_calls.get('function', []) + noise.get('function', [])) | set(same_path.get('function', [])) | set(external_entity.get('function', []))),
                'variable': list(set(outgoing_calls.get('variable', []) + noise.get('variable', [])) | set(same_path.get('variable', [])) | set(external_entity.get('variable', []))),
                'segment': list(set(target_comp.get('candidate_segment', [])) | set(same_path.get('segment', []))),
            }
            
            candidate_dict = {
                'class': self.process_candidates(target_comp, components, all_candidates['class'], 'class', repo_dir),
                'function': self.process_candidates(target_comp, components, all_candidates['function'], 'function', repo_dir),
                'variable': self.process_candidates(target_comp, components, all_candidates['variable'], 'variable', repo_dir),
                'segment': self.process_candidates(target_comp, components, all_candidates['segment'], 'segment', repo_dir)
            }
            
            class_candidate_ids = list(candidate_dict['class'].keys())
            method_candidate_ids = self.get_method_candidates_from_classes(class_candidate_ids, components)
            candidate_dict['method'] = self.process_candidates(target_comp, components, method_candidate_ids, 'method', repo_dir)
            
            if not self.extend:
                for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
                    candidate_dict[comp_type] = {
                        cand_id: cand_info 
                        for cand_id, cand_info in candidate_dict[comp_type].items() 
                        if not cand_info.get('is_third_party', False)
                    }

            self.filter_references_to_candidates(candidate_dict)

            import_statements = self.get_import_statements(external_knowledge, target_relative_path)
            full_name = target_component_id
            clean_name = target_component_id.split('@')[0] if '@' in target_component_id else target_component_id

            processed_sample = {
                'id': task_id,
                'repo_name': repo_name,
                'target_function_prompt': target_function_prompt,
                'target_method_prompt': " ",
                'relative_path': target_relative_path,
                'full_name': full_name,
                'clean_name': clean_name,
                'implementation_plan': implementation_plan,
                "type": "function",
                'candidate': candidate_dict,
                'import_statements': import_statements,
                'g_dependency': g_dependency
            }
            
            return processed_sample
            
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            return None
    
    def get_output_filename(self) -> str:
        if self.model:
            return f"processed_RepoExec_{self.model}.jsonl"
        return "processed_RepoExec.jsonl"


class DevEval_Loader(Benchmark_Loader):
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None,
                 data_path: str = None, extend: bool = False, model: str = None, candidate_only: bool = False):
        super().__init__(graphs_base_dir, output_dir, model, candidate_only)
        if data_path is None:
            repo_root = Path(__file__).resolve().parents[1]
            self.data_path = os.path.join(str(repo_root), 'benchmark', 'DevEval', 'data.jsonl')
        else:
            self.data_path = data_path
        self.extend = extend
    
    def extract_repo_name(self, project_path: str) -> str:
        """Extract repository name from project_path field."""
        return project_path.split('/')[-1]
    
    def construct_component_id_from_namespace(self, namespace: str, completion_path: str, sample_type: str, repo_name: str) -> List[str]:
        """Construct component ID from namespace and completion_path."""  
        namespace_parts = namespace.split('.')
        
        if sample_type == "function":
            function_name = namespace_parts[-1]
            path_parts = namespace_parts[:-1]
            
            primary_ids = []
            if completion_path:
                path_parts_from_completion = completion_path.split('/')
                if len(path_parts_from_completion) >= 2:
                    relative_path = '/'.join(path_parts_from_completion[1:])
                    primary_ids.extend([
                        f"{function_name}@{relative_path}",
                        f"{function_name}@src/{relative_path}",
                        f"{function_name}@{repo_name}/{relative_path}",
                        f"{function_name}@{repo_name}/src/{relative_path}"
                    ])
            
            possible_ids = []
            if path_parts:
                file_path = '/'.join(path_parts) + '.py'
                init_path = '/'.join(path_parts) + '/__init__.py'
                
                possible_ids.extend([
                    f"{function_name}@{file_path}",
                    f"{function_name}@{init_path}",
                    f"{function_name}@src/{file_path}",
                    f"{function_name}@src/{init_path}",
                    f"{function_name}@{repo_name}/{file_path}",
                    f"{function_name}@{repo_name}/{init_path}",
                    f"{function_name}@{repo_name}/src/{file_path}",
                    f"{function_name}@{repo_name}/src/{init_path}"
                ])
            else:
                path_parts_from_completion = completion_path.split('/')
                if len(path_parts_from_completion) >= 3:
                    file_path = '/'.join(path_parts_from_completion[2:])
                else:
                    file_path = path_parts_from_completion[-1]
                
                possible_ids.extend([
                    f"{function_name}@{file_path}",
                    f"{function_name}@src/{file_path}",
                    f"{function_name}@{repo_name}/{file_path}",
                    f"{function_name}@{repo_name}/src/{file_path}"
                ])
            
            return primary_ids + possible_ids
            
        elif sample_type == "method":
            if len(namespace_parts) >= 2:
                method_part = '.'.join(namespace_parts[-2:])
                path_parts = namespace_parts[:-2]
                
                primary_ids = []
                if completion_path:
                    path_parts_from_completion = completion_path.split('/')
                    if len(path_parts_from_completion) >= 2:
                        relative_path = '/'.join(path_parts_from_completion[1:])
                        primary_ids.extend([
                            f"{method_part}@{relative_path}",
                            f"{method_part}@src/{relative_path}",
                            f"{method_part}@{repo_name}/{relative_path}",
                            f"{method_part}@{repo_name}/src/{relative_path}"
                        ])
                
                possible_ids = []
                if path_parts:
                    file_path = '/'.join(path_parts) + '.py'
                    init_path = '/'.join(path_parts) + '/__init__.py'
                else:
                    file_path = namespace_parts[0] + '.py'
                    init_path = namespace_parts[0] + '/__init__.py'
                
                possible_ids.extend([
                    f"{method_part}@{file_path}",
                    f"{method_part}@src/{file_path}",
                    f"{method_part}@{init_path}",
                    f"{method_part}@src/{init_path}",
                    f"{method_part}@{repo_name}/{file_path}",
                    f"{method_part}@{repo_name}/src/{file_path}",
                    f"{method_part}@{repo_name}/{init_path}",
                    f"{method_part}@{repo_name}/src/{init_path}"
                ])
                
                func_name = method_part.split('.', 1)[1] if '.' in method_part else method_part
                possible_ids.extend([
                    f"{func_name}@{file_path}",
                    f"{func_name}@src/{file_path}",
                    f"{func_name}@{init_path}",
                    f"{func_name}@src/{init_path}",
                    f"{func_name}@{repo_name}/{file_path}",
                    f"{func_name}@{repo_name}/src/{file_path}",
                    f"{func_name}@{repo_name}/{init_path}",
                    f"{func_name}@{repo_name}/src/{init_path}"
                ])
                
                if completion_path:
                    path_parts_from_completion = completion_path.split('/')
                    if len(path_parts_from_completion) >= 2:
                        relative_path = '/'.join(path_parts_from_completion[1:])
                        possible_ids.extend([
                            f"{func_name}@{relative_path}",
                            f"{func_name}@src/{relative_path}",
                            f"{func_name}@{repo_name}/{relative_path}",
                            f"{func_name}@{repo_name}/src/{relative_path}"
                        ])
                
                return primary_ids + possible_ids
        
        return []
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load DevEval dataset from local JSONL file."""
        try:
            print(f"Loading DevEval dataset from {self.data_path}...")
            samples = []
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
                        continue
            
            print(f"Loaded {len(samples)} samples from DevEval")
            return samples
            
        except Exception as e:
            print(f"Error loading DevEval dataset: {e}")
            return None
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single DevEval sample and create the required format."""
        try:
            namespace = sample.get('namespace', '')
            sample_type = sample.get('type', '')
            project_path = sample.get('project_path', '')
            completion_path = sample.get('completion_path', '')
            target_function_prompt = sample.get('target_function_prompt', '')
            target_method_prompt = sample.get('target_method_prompt', '')
            implementation_plan = sample.get('implementation_plan', [])

            task_id = namespace
            
            repo_name = self.extract_repo_name(project_path)
            graph_path = self.find_dependency_graph_path(repo_name)
            
            if not graph_path:
                print(f"Warning: No dependency graph found for repo {repo_name}")
                return None
            
            components = self.load_dependency_graph(graph_path)
            if not components:
                print(f"Warning: Empty dependency graph for repo {repo_name}")
                return None
            
            repo_dir = os.path.dirname(graph_path)
            external_knowledge = self.load_external_knowledge(repo_dir)
            
            target_component_ids = self.construct_component_id_from_namespace(namespace, completion_path, sample_type, repo_name)
            
            if not target_component_ids:
                print(f"Warning: Could not construct component IDs for {namespace}")
                return None
            
            target_component_id = None
            for possible_id in target_component_ids:
                if possible_id in components:
                    target_component_id = possible_id
                    break
            
            if not target_component_id:
                print(f"Warning: Target component not found for {namespace}")
                print(f"Checked IDs:", target_component_ids)
                return None
            
            target_comp = components[target_component_id]
            target_relative_path = target_comp.get('relative_path', '')
            
            outgoing_calls = target_comp.get('outgoing_calls', {"class": [], "function": [], "method": [], "variable": []})
            noise = target_comp.get('noise', {"class": [], "function": [], "method": [], "variable": []})
            g_dependency = list(dict.fromkeys(
                list(outgoing_calls.get('class', [])) +
                list(outgoing_calls.get('function', [])) +
                list(outgoing_calls.get('variable', []))
            ))
            
            same_path = self.get_same_path_candidate_ids(components, target_relative_path, target_component_id)
            external_entity = self.get_external_entity_candidate_ids(
                external_knowledge, target_relative_path, target_component_id, components
            )
            all_candidates = {
                'class': list(set(outgoing_calls.get('class', []) + noise.get('class', [])) | set(same_path.get('class', [])) | set(external_entity.get('class', []))),
                'function': list(set(outgoing_calls.get('function', []) + noise.get('function', [])) | set(same_path.get('function', [])) | set(external_entity.get('function', []))),
                'variable': list(set(outgoing_calls.get('variable', []) + noise.get('variable', [])) | set(same_path.get('variable', [])) | set(external_entity.get('variable', []))),
                'segment': list(set(target_comp.get('candidate_segment', [])) | set(same_path.get('segment', []))),
            }
            
            candidate_dict = {
                'class': self.process_candidates(target_comp, components, all_candidates['class'], 'class', repo_dir),
                'function': self.process_candidates(target_comp, components, all_candidates['function'], 'function', repo_dir),
                'variable': self.process_candidates(target_comp, components, all_candidates['variable'], 'variable', repo_dir),
                'segment': self.process_candidates(target_comp, components, all_candidates['segment'], 'segment', repo_dir)
            }
            
            class_candidate_ids = list(candidate_dict['class'].keys())
            method_candidate_ids = self.get_method_candidates_from_classes(class_candidate_ids, components)
            candidate_dict['method'] = self.process_candidates(target_comp, components, method_candidate_ids, 'method', repo_dir)
            
            if not self.extend:
                for comp_type in ['class', 'function', 'method', 'variable', 'segment']:
                    candidate_dict[comp_type] = {
                        cand_id: cand_info 
                        for cand_id, cand_info in candidate_dict[comp_type].items() 
                        if not cand_info.get('is_third_party', False)
                    }

            self.filter_references_to_candidates(candidate_dict)

            import_statements = self.get_import_statements(external_knowledge, target_relative_path)
            full_name = target_component_id
            clean_name = target_component_id.split('@')[0] if '@' in target_component_id else target_component_id

            processed_sample = {
                'id': task_id,
                'repo_name': repo_name,
                'target_function_prompt': target_function_prompt,
                'target_method_prompt': target_method_prompt,
                'relative_path': target_relative_path,
                'full_name': full_name,
                'clean_name': clean_name,
                'implementation_plan': implementation_plan,
                'type': sample_type,
                'candidate': candidate_dict,
                'import_statements': import_statements,
                'g_dependency': g_dependency
            }
            
            return processed_sample
            
        except Exception as e:
            print(f"Error processing sample {sample.get('namespace', 'unknown')}: {e}")
            return None
    
    def get_output_filename(self) -> str:
        if self.model:
            return f"processed_DevEval_{self.model}.jsonl"
        return "processed_DevEval.jsonl"


class VersionExec_Loader(Benchmark_Loader):
    
    LIB_NAME_MAPPING = {
        'sklearn': ['scikit-learn', 'sklearn'],
        'skimage': ['scikit-image', 'skimage'],
        'cv2': ['opencv-python', 'cv2'],
        'yaml': ['PyYAML', 'yaml'],
    }
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None, data_path: str = None, extend: bool = False, model: str = None, benchmark_type: str = "old", candidate_only: bool = False):
        super().__init__(graphs_base_dir, output_dir, model, candidate_only)
        self.extend = extend
        self.benchmark_type = benchmark_type
        if data_path is None:
            repo_root = Path(__file__).resolve().parents[1]
            if benchmark_type == "old":
                self.data_path = os.path.join(str(repo_root), 'benchmark', 'versionexec_old', 'versionexec_old.jsonl')
            else:
                self.data_path = os.path.join(str(repo_root), 'benchmark', 'versionexec_new', 'versionexec_new.jsonl')
        else:
            self.data_path = data_path
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        try:
            print(f"Loading dataset from {self.data_path}...")
            dataset = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dataset.append(json.loads(line))
            print(f"Loaded {len(dataset)} samples from VersionExec")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            task_id = sample['task_id']
            complete_prompt = sample.get('complete_prompt', '')
            libs_raw = sample.get('libs', '[]')
            
            if isinstance(libs_raw, str):
                import ast
                try:
                    libs = ast.literal_eval(libs_raw)
                except:
                    libs = []
            else:
                libs = libs_raw if isinstance(libs_raw, list) else []
            
            all_candidates = {'class': {}, 'function': {}}
            
            for lib in libs:
                if lib in self.LIB_NAME_MAPPING:
                    possible_names = self.LIB_NAME_MAPPING[lib]
                else:
                    possible_names = [lib]
                
                graph_path = None
                actual_lib_name = None
                for possible_name in possible_names:
                    lib_dir = os.path.join(self.graphs_base_dir, possible_name)
                    test_graph_path = os.path.join(lib_dir, "dependency_graph.json")
                    
                    if os.path.exists(test_graph_path):
                        graph_path = test_graph_path
                        actual_lib_name = possible_name
                        break
                
                if graph_path is None:
                    continue 
                
                components = self.load_dependency_graph(graph_path)
                
                for comp_id, comp_data in components.items():
                    comp_type = comp_data.get('component_type', '')
                    
                    if comp_type not in ['class', 'function']:
                        continue
                    if (
                        comp_id.startswith('_') or
                        '@_' in comp_id or
                        '/_' in comp_id
                    ):
                        continue
                    
                    docstring = comp_data.get('docstring', '')
                    all_candidates[comp_type][comp_id] = {
                        'relative_path': comp_data.get('relative_path', ''),
                        'source_code': comp_data.get('source_code', ''),
                        'docstring': docstring,
                        'description': docstring,
                        'lib_name': actual_lib_name,
                        'is_third_party': True,
                        'library': actual_lib_name,
                    }
            
            processed_sample = {
                'id': task_id,
                'target_function_prompt': complete_prompt,
                'target_method_prompt': " ",
                'relative_path': "",
                'type': "function",
                'candidate': all_candidates,
                'import_statements': []
            }
            
            return processed_sample
            
        except Exception as e:
            print(f"Error processing sample {sample.get('task_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_output_filename(self) -> str:
        base_name = f"processed_versionexec_{self.benchmark_type}"
        if self.model:
            return f"{base_name}_{self.model}.jsonl"
        return f"{base_name}.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Process a benchmark and generate processed outputs.")
    parser.add_argument('--benchmark', choices=['RepoExec', 'DevEval', 'versionexec_old', 'versionexec_new'], help='Benchmark to process')
    parser.add_argument('--extend', type=lambda x: x.lower() == 'true', default=True, help='Include third-party candidates (True/False, default: True)')
    parser.add_argument('--model', type=str, required=True,
                        help='Direct model name used for output file naming.')
    parser.add_argument('--candidate_only', action='store_true', help='Reload only candidate field from dependency graphs, keep all other fields from existing processed file')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.benchmark in ['versionexec_old', 'versionexec_new']:
        graphs_base_dir = os.path.join(str(repo_root), 'data', 'parser_output', args.benchmark)
    else:
        graphs_base_dir = os.path.join(str(repo_root), 'data', 'parser_output', args.benchmark)

    output_dir = os.path.join(str(repo_root), 'data', 'processed_benchmarks')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Processing {args.benchmark} Dataset")
    print(f"Extend mode: {args.extend}")
    print(f"Model: {args.model}")
    print("=" * 60)

    if args.benchmark == 'RepoExec':
        loader = RepoExec_Loader(
            graphs_base_dir=graphs_base_dir,
            output_dir=output_dir,
            extend=args.extend,
            model=args.model,
            candidate_only=args.candidate_only
        )
    elif args.benchmark == 'DevEval':
        loader = DevEval_Loader(
            graphs_base_dir=graphs_base_dir,
            output_dir=output_dir,
            extend=args.extend,
            model=args.model,
            candidate_only=args.candidate_only
        )
    elif args.benchmark == 'versionexec_old':
        loader = VersionExec_Loader(
            graphs_base_dir=graphs_base_dir,
            output_dir=output_dir,
            extend=args.extend,
            model=args.model,
            benchmark_type="old",
            candidate_only=args.candidate_only
        )
    else:
        loader = VersionExec_Loader(
            graphs_base_dir=graphs_base_dir,
            output_dir=output_dir,
            extend=args.extend,
            model=args.model,
            benchmark_type="new",
            candidate_only=args.candidate_only
        )
    
    dataset = loader.load_dataset()
    
    loader.process_dataset()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
