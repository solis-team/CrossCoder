import os
import re
import json
from pathlib import Path
from argparse import ArgumentParser


def retrieve_requirement_file(project_path: str):
    """Return all requirement .txt files found in the root or in the requirements folder."""
    requirement_files = []
    if os.path.exists(os.path.join(project_path, 'requirements.txt')):
        requirement_files.append('requirements.txt')
    if os.path.exists(os.path.join(project_path, 'requirements')):
        requirement_dir = os.path.join(project_path, 'requirements')
        for file in os.listdir(requirement_dir):
            if file.endswith('.txt'):
                requirement_files.append(os.path.join('requirements', file))
    return requirement_files


def parse_requirements_txt(file_path: str):
    """Parse a requirements.txt-like file and return a list of dependencies."""
    requirements = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements


def parse_setup_py(file_path: str):
    """Extract install_requires content from a setup.py file using regex heuristics."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    pattern_list = re.findall(r'install_requires\s*=\s*\[([^\]]*)\]', content, re.DOTALL)
    deps = []
    for block in pattern_list:
        matches = re.findall(r'["\']([A-Za-z0-9_.\-]+[^"\']*)["\']', block)
        deps.extend(matches)
    if not deps:
        match_var = re.search(r'install_requires\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', content)
        if match_var:
            var_name = match_var.group(1)
            var_block = re.search(rf'{var_name}\s*=\s*\[([^\]]*)\]', content, re.DOTALL)
            if var_block:
                matches = re.findall(r'["\']([A-Za-z0-9_.\-]+[^"\']*)["\']', var_block.group(1))
                deps.extend(matches)
    return deps


def extract_requirements_repoexec(repo_path: Path):
    """Extract dependencies from package.txt for RepoExec repos."""
    package_txt = repo_path / 'package.txt'
    if not package_txt.exists():
        return None
    return parse_requirements_txt(str(package_txt))


def extract_requirements_deveval(repo_path: Path):
    """Extract dependencies from requirements files and setup.py for DevEval repos."""
    requirements = []
    for req_file in retrieve_requirement_file(str(repo_path)):
        full_path = os.path.join(repo_path, req_file)
        requirements.extend(parse_requirements_txt(full_path))
    setup_path = os.path.join(repo_path, 'setup.py')
    if os.path.exists(setup_path):
        requirements.extend(parse_setup_py(setup_path))
    return list(sorted(set(requirements)))


def save_requirements(output_path: Path, dependencies):
    """Save dependencies to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dependencies, f, indent=2, ensure_ascii=False)


def process_repoexec():
    """Process RepoExec dataset and extract requirements for each valid repo."""
    base_path = Path('benchmark/RepoExec/test-apps')
    output_root = Path('data/parser_output/RepoExec')
    for repo in base_path.iterdir():
        if not repo.is_dir():
            continue
        deps = extract_requirements_repoexec(repo)
        if not deps:
            continue
        save_requirements(output_root / repo.name / 'requirements.json', deps)


def process_deveval():
    """Process DevEval dataset and extract requirements for each repo under all topics."""
    base_path = Path('benchmark/DevEval/Source_Code')
    output_root = Path('data/parser_output/DevEval')
    for topic in base_path.iterdir():
        if not topic.is_dir():
            continue
        for repo in topic.iterdir():
            if not repo.is_dir():
                continue
            deps = extract_requirements_deveval(repo)
            if deps:
                save_requirements(output_root / repo.name / 'requirements.json', deps)


def main():
    """Main entrypoint for extracting dependencies based on dataset type."""
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['RepoExec', 'DevEval'], required=True)
    args = parser.parse_args()
    if args.dataset == 'RepoExec':
        process_repoexec()
    else:
        process_deveval()


if __name__ == '__main__':
    main()