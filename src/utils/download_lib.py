import json
import os
import requests
import re
import tarfile
import argparse
import shutil
from packaging import version
from packaging.specifiers import SpecifierSet


def extract_name_version(dep_str: str):
    """
    Extract the package name and version specification from a dependency string.
    Handles complex package names with hyphens, underscores, and dots.
    """
    dep_str = dep_str.strip()
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9\-_.]*[A-Za-z0-9]|[A-Za-z0-9])\s*(.*)$", dep_str)
    if match:
        name = match.group(1).replace("_", "-").lower()
        version_spec = match.group(2).strip()
        return name, version_spec
    return None, None


def normalize_version(ver: str):
    """
    Normalize version string for comparison by removing local version identifiers
    and handling development versions.
    """
    ver = re.sub(r'\+.*$', '', ver)
    ver = re.sub(r'\.dev\d*$', '.dev0', ver)
    return ver


def is_version_compatible(ver: str, version_spec: str):
    """
    Check if a version satisfies the version specification using packaging library.
    """
    if not version_spec:
        return True
    try:
        normalized_ver = normalize_version(ver)
        spec_set = SpecifierSet(version_spec)
        return version.parse(normalized_ver) in spec_set
    except Exception:
        return False


def get_latest_compatible_version(name: str, version_spec: str):
    """
    Get all compatible versions of a package from PyPI that satisfy the version specification.
    Returns a list sorted from newest to oldest.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        all_versions = list(data["releases"].keys())        
        valid_versions = []
        for ver in all_versions:
            if data["releases"][ver]: 
                valid_versions.append(ver)
        
        if not valid_versions:
            return []
        
        sorted_versions = sorted(valid_versions, key=lambda v: version.parse(normalize_version(v)))
        compatible_versions = []
        for ver in reversed(sorted_versions):
            if is_version_compatible(ver, version_spec):
                compatible_versions.append(ver)
        
        return compatible_versions
    except Exception as e:
        print(f"Error fetching versions for {name}: {e}")
        return []


def download_package(name: str, pkg_version: str, target_dir: str):
    import subprocess
    
    try:
        package_spec = f"{name}=={pkg_version}"
        print(f"Downloading {package_spec} using pip...")

        result = subprocess.run([
            "pip", "download", 
            "--only-binary=:all:", 
            "--no-deps", 
            package_spec
        ], cwd=target_dir, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Wheel not available, trying source distribution...")
            result = subprocess.run([
                "pip", "download", 
                "--no-deps", 
                package_spec
            ], cwd=target_dir, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"Failed to download {package_spec}: {result.stderr}")
                return False
        
        downloaded_files = [f for f in os.listdir(target_dir) if f.endswith(('.whl', '.tar.gz', '.zip'))]
        
        if not downloaded_files:
            print(f"No package files found for {package_spec}")
            return False
        
        print(f"Successfully downloaded {package_spec}")
        return True
        
    except Exception as e:
        print(f"Error downloading {name}=={pkg_version}: {e}")
        return False


def download_package_latest(name: str, target_dir: str):
    import subprocess
    
    try:
        print(f"Downloading latest {name} using pip...")

        result = subprocess.run([
            "pip", "download", 
            "--only-binary=:all:", 
            "--no-deps", 
            name
        ], cwd=target_dir, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Wheel not available, trying source distribution...")
            result = subprocess.run([
                "pip", "download", 
                "--no-deps", 
                name
            ], cwd=target_dir, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"Failed to download latest {name}: {result.stderr}")
                return False
        
        downloaded_files = [f for f in os.listdir(target_dir) if f.endswith(('.whl', '.tar.gz', '.zip'))]
        
        if not downloaded_files:
            print(f"No package files found for latest {name}")
            return False
        
        print(f"Successfully downloaded latest {name}")
        return True
        
    except Exception as e:
        print(f"Error downloading latest {name}: {e}")
        return False


def wrap_repos_for_benchmark(benchmark: str):
    """
    Wrap all existing repositories by creating a subfolder with the same name.
    Creates a copy of the original directory and wraps that instead, leaving original untouched.
    """
    if benchmark == "RepoExec":
        original_dir = "benchmark/RepoExec/test-apps"
        benchmark_dir = "benchmark/RepoExec/test-apps-copy"
    elif benchmark == "DevEval":
        original_dir = "benchmark/DevEval/Source_Code"
        benchmark_dir = "benchmark/DevEval/SourceCode-copy"
    else:
        print(f"Unknown benchmark type: {benchmark}")
        return

    if not os.path.exists(original_dir):
        print(f"Original benchmark directory not found: {original_dir}")
        return

    if not os.path.exists(benchmark_dir):
        print(f"Creating copy: {original_dir} -> {benchmark_dir}")
        try:
            shutil.copytree(original_dir, benchmark_dir)
            print(f"Copy created successfully")
        except Exception as e:
            print(f"Error creating copy: {e}")
            return

    print(f"Wrapping repositories for {benchmark} in {benchmark_dir}...")
    
    if benchmark == "RepoExec":
        repos = [d for d in os.listdir(benchmark_dir) 
                if os.path.isdir(os.path.join(benchmark_dir, d))]
        
        for repo_name in repos:
            repo_path = os.path.join(benchmark_dir, repo_name)
            nested_repo_path = os.path.join(repo_path, repo_name)
            
            temp_path = repo_path + "_temp"
            try:
                shutil.move(repo_path, temp_path)
                os.makedirs(repo_path, exist_ok=True)
                shutil.move(temp_path, nested_repo_path)
                print(f"  Wrapped {repo_name}")
            except Exception as e:
                print(f"  Error wrapping {repo_name}: {e}")
                if os.path.exists(temp_path):
                    shutil.move(temp_path, repo_path)
                    
    elif benchmark == "DevEval":
        topic_dirs = ["Communications", "Database", "Internet", "Multimedia", 
                     "Scientific-Engineering", "Security", "Software-Development", 
                     "System", "Text-Processing", "Utilities"]
        
        for topic in topic_dirs:
            topic_path = os.path.join(benchmark_dir, topic)
            if not os.path.exists(topic_path):
                continue
                
            repos = [d for d in os.listdir(topic_path) 
                    if os.path.isdir(os.path.join(topic_path, d))]
            
            for repo_name in repos:
                repo_path = os.path.join(topic_path, repo_name)
                nested_repo_path = os.path.join(repo_path, repo_name)
                
                temp_path = repo_path + "_temp"
                try:
                    shutil.move(repo_path, temp_path)
                    os.makedirs(repo_path, exist_ok=True)
                    shutil.move(temp_path, nested_repo_path)
                    print(f"  Wrapped {topic}/{repo_name}")
                except Exception as e:
                    print(f"  Error wrapping {topic}/{repo_name}: {e}")
                    if os.path.exists(temp_path):
                        shutil.move(temp_path, repo_path)


def process_repo_dependencies(benchmark: str, repo_name: str):
    """
    Process a single repository by downloading its dependencies.
    """
    requirements_file = f"data/parser_output/{benchmark}/{repo_name}/requirements.json"

    if not os.path.exists(requirements_file):
        print(f"Requirements file not found: {requirements_file}")
        return
    
    if benchmark == "RepoExec":
        # Download to inner wrapped folder
        target_dir = f"benchmark/RepoExec/test-apps-copy/{repo_name}/{repo_name}"
    elif benchmark == "DevEval":
        topic_dirs = ["Communications", "Database", "Internet", "Multimedia",
                     "Scientific-Engineering", "Security", "Software-Development",
                     "System", "Text-Processing", "Utilities"]
        target_dir = None
        for topic in topic_dirs:
            # Download to inner wrapped folder
            potential_dir = f"benchmark/DevEval/SourceCode-copy/{topic}/{repo_name}/{repo_name}"
            if os.path.exists(potential_dir):
                target_dir = potential_dir
                break
        if not target_dir:
            print(f"Could not find target directory for DevEval repo: {repo_name}")
            return
    else:
        print(f"Unknown benchmark type: {benchmark}")
        return
    
    try:
        with open(requirements_file, "r") as f:
            deps = json.load(f)
    except Exception as e:
        print(f"Error reading requirements file {requirements_file}: {e}")
        return
    
    print(f"Processing {repo_name} ({len(deps)} dependencies)...")
    
    for dep in deps:
        name, version_spec = extract_name_version(dep)
        if not name:
            print(f"Could not parse dependency: {dep}")
            continue
        
        print(f"  Finding compatible versions for {name} {version_spec}")
        compatible_versions = get_latest_compatible_version(name, version_spec)
        
        if not compatible_versions:
            print(f"  No compatible versions found for {name} {version_spec}")
            continue
        
        downloaded = False
        for version_to_try in compatible_versions:
            try:
                print(f"  Trying to download {name}=={version_to_try}")
                if download_package(name, version_to_try, target_dir):
                    downloaded = True
                    break
            except Exception as e:
                print(f"  Failed to download {name}=={version_to_try}: {e}")
                continue
        
        if not downloaded:
            try:
                print(f"  Fallback: trying to download latest {name}")
                if download_package_latest(name, target_dir):
                    downloaded = True
            except Exception as e:
                print(f"  Fallback failed for {name}: {e}")
        
        if not downloaded:
            print(f"  Could not download any version of {name}")
    
    print(f"Extracting all package files for {repo_name}...")
    import zipfile
    
    package_files = [f for f in os.listdir(target_dir) if f.endswith(('.whl', '.tar.gz', '.zip'))]
    
    for package_file in package_files:
        package_path = os.path.join(target_dir, package_file)
        try:
            if package_file.endswith('.whl') or package_file.endswith('.zip'):
                with zipfile.ZipFile(package_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove(package_path)
            elif package_file.endswith('.tar.gz'):
                with tarfile.open(package_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir)
                os.remove(package_path)
        except Exception as e:
            print(f"  Error extracting {package_file}: {e}")
    
    print(f"Completed processing {repo_name}")


def download_dependencies_for_benchmark(benchmark: str):
    if benchmark in ["versionexec_old", "versionexec_new"]:
        requirements_file = f"benchmark/{benchmark}/requirements.json"
        target_dir = f"benchmark/{benchmark}/third_party"
        
        if not os.path.exists(requirements_file):
            print(f"Requirements file not found: {requirements_file}")
            return
        
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            with open(requirements_file, "r") as f:
                libs = json.load(f)
        except Exception as e:
            print(f"Error reading requirements file: {e}")
            return
        
        print(f"Downloading {len(libs)} packages for {benchmark}...")
        
        for lib_spec in libs:
            name, version_spec = extract_name_version(lib_spec)
            if not name:
                print(f"Could not parse: {lib_spec}")
                continue
            
            if version_spec and version_spec.startswith("=="):
                pkg_version = version_spec[2:]
                print(f"Downloading {name}=={pkg_version}")
                download_package(name, pkg_version, target_dir)
            else:
                print(f"Downloading latest {name}")
                download_package_latest(name, target_dir)
        
        print("Extracting package files...")
        import zipfile
        
        package_files = [f for f in os.listdir(target_dir) if f.endswith(('.whl', '.tar.gz', '.zip'))]
        
        for package_file in package_files:
            package_path = os.path.join(target_dir, package_file)
            try:
                if package_file.endswith('.whl') or package_file.endswith('.zip'):
                    with zipfile.ZipFile(package_path, 'r') as zip_ref:
                        zip_ref.extractall(target_dir)
                    os.remove(package_path)
                elif package_file.endswith('.tar.gz'):
                    with tarfile.open(package_path, 'r:gz') as tar_ref:
                        tar_ref.extractall(target_dir)
                    os.remove(package_path)
            except Exception as e:
                print(f"Error extracting {package_file}: {e}")
        
        print(f"Completed {benchmark} downloads")
        return
    
    parser_output_dir = f"data/parser_output/{benchmark}"
    
    if not os.path.exists(parser_output_dir):
        print(f"Parser output directory not found: {parser_output_dir}")
        return
    
    repos = [d for d in os.listdir(parser_output_dir) 
             if os.path.isdir(os.path.join(parser_output_dir, d))]
    
    print(f"Found {len(repos)} repositories in {benchmark}")
    
    for repo_name in repos:
        process_repo_dependencies(benchmark, repo_name)


def main():
    parser = argparse.ArgumentParser(description="Process package dependencies for benchmark repositories")
    parser.add_argument("--benchmark", required=True, choices=["RepoExec", "DevEval", "versionexec_old", "versionexec_new"],
                       help="Benchmark type (RepoExec, DevEval, versionexec_old, or versionexec_new)")
    parser.add_argument("--wrap", action="store_true",
                       help="Wrap repositories by creating nested folders")
    parser.add_argument("--download", action="store_true", 
                       help="Download third-party dependencies")
    
    args = parser.parse_args()
    
    if not args.wrap and not args.download:
        print("Error: Must specify at least one operation (--wrap or --download)")
        return
    
    benchmark = args.benchmark
    
    if args.wrap:
        if benchmark in ["versionexec_old", "versionexec_new"]:
            print(f"{benchmark} does not require wrapping")
        else:
            print("Step 1: Wrapping repositories...")
            wrap_repos_for_benchmark(benchmark)
    
    if args.download:
        print("Step 2: Downloading third-party dependencies...")
        download_dependencies_for_benchmark(benchmark)


if __name__ == "__main__":
    main()
