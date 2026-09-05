#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from bigcodebench.sanitize import sanitize
from bigcodebench.data import get_bigcodebench


def normalize_task_id(task_id):
    if isinstance(task_id, int):
        return task_id
    task_id = str(task_id)
    if "/" in task_id and not task_id.startswith("versionexec/"):
        return "versionexec/" + task_id.split("/", 1)[1]
    return task_id


def convert_generated_to_bcb(generated_file, benchmark_file, output_file, folder_name):
    dataset = get_bigcodebench()
    
    entry_point = {}
    task_id_map = {}
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                sample = json.loads(line)
                task_id_str = sample.get('task_id')
                entry_point_name = sample.get('entry_point')
                if task_id_str and entry_point_name:
                    task_id_map[idx] = task_id_str
                    entry_point[task_id_str] = entry_point_name
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    converted_samples = []
    with open(generated_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            sample = json.loads(line)
            task_id = sample.get('task_id')
            responses = sample.get('response', [])
            
            if not isinstance(responses, list):
                responses = [responses]
            
            if isinstance(task_id, int):
                task_id_str = task_id_map.get(task_id)
                if not task_id_str:
                    print(f"Warning: task_id {task_id} not found in benchmark file, skipping")
                    continue
            else:
                task_id_str = normalize_task_id(task_id)
            
            function_name = entry_point.get(task_id_str)
            if not function_name:
                print(f"Warning: entry_point not found for task_id {task_id_str}, skipping")
                continue
            
            for response in responses:
                if not response:
                    continue
                
                sanitized_code = sanitize(code=response, entrypoint=function_name)
                
                converted_sample = {
                    "task_id": task_id_str,
                    "solution": sanitized_code
                }
                converted_samples.append(converted_sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_samples)} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert generated file to BigCodeBench format")
    parser.add_argument("--generated_file", type=str, required=True, help="Path to generated JSONL file")
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_file = os.path.join(str(repo_root), 'benchmark', 'versionexec_new', 'versionexec_new.jsonl')
    
    generated_path = Path(args.generated_file)
    folder_name = generated_path.parent.name
    
    result_dir = os.path.join(str(repo_root), 'benchmark', 'versionexec_new', 'bcb_results')
    output_file = os.path.join(result_dir, f"{folder_name}_sanitized_calibrated.jsonl")
    
    convert_generated_to_bcb(args.generated_file, benchmark_file, output_file, folder_name)
