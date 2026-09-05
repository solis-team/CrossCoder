#!/bin/bash
set -e  

DATASET=""
REPOS_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_BASE_DIR="$ROOT_DIR/data/parser_output"

usage() {
    echo "Usage: $0 --dataset [RepoExec|DevEval|versionexec_old|versionexec_new] [--repos_dir path]"
    echo ""
    echo "Arguments:"
    echo "  --dataset     Dataset type: RepoExec, DevEval, versionexec_old, or versionexec_new (required)"
    echo "  --repos_dir   Path to repositories directory (optional)"
    echo ""
    echo "Default repos_dir:"
    echo "  RepoExec:          benchmark/RepoExec/test-apps"
    echo "  DevEval:           benchmark/DevEval/Source_Code"
    echo "  versionexec_old:  benchmark/versionexec_old/third_party"
    echo "  versionexec_new:  benchmark/versionexec_new/third_party"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset RepoExec"
    echo "  $0 --dataset DevEval --repos_dir /path/to/deveval/repos"
    echo "  $0 --dataset versionexec_old"
    echo "  $0 --dataset versionexec_new"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --repos_dir)
            REPOS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset argument is required"
    usage
fi

if [[ "$DATASET" != "RepoExec" && "$DATASET" != "DevEval" && "$DATASET" != "versionexec_old" && "$DATASET" != "versionexec_new" ]]; then
    echo "Error: Dataset must be either 'RepoExec', 'DevEval', 'versionexec_old', or 'versionexec_new'"
    usage
fi


if [[ -z "$REPOS_DIR" ]]; then
    if [[ "$DATASET" == "RepoExec" ]]; then
        REPOS_DIR="$ROOT_DIR/benchmark/RepoExec/test-apps"
    elif [[ "$DATASET" == "DevEval" ]]; then
        REPOS_DIR="$ROOT_DIR/benchmark/DevEval/Source_Code"
    elif [[ "$DATASET" == "versionexec_old" ]]; then
        REPOS_DIR="$ROOT_DIR/benchmark/versionexec_old/third_party"
    else
        REPOS_DIR="$ROOT_DIR/benchmark/versionexec_new/third_party"
    fi
fi

REPOS_DIR="$(cd "$REPOS_DIR" 2>/dev/null && pwd)" || {
    echo "Error: Repository directory '$REPOS_DIR' does not exist"
    exit 1
}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Repos directory: $REPOS_DIR"
echo "  Output directory: $OUTPUT_BASE_DIR/$DATASET"
echo ""

OUTPUT_DIR="$OUTPUT_BASE_DIR/$DATASET"
mkdir -p "$OUTPUT_DIR"

is_repo_processed() {
    local output_path="$1"
    
    # if [[ ! -f "$output_path/requirements.json" ]]; then
    #     return 0
    # fi
    
    if [[ -f "$output_path/dependency_graph.json" && -f "$output_path/external_knowledge.json" ]]; then
        return 0
    else
        return 1
    fi
}

normalize_package_name() {
    local name="$1"
    echo "$name" | sed -E 's/-[0-9]+(\.[0-9]+)*(\.[a-z]+[0-9]*)?$//'
}

run_ast_parser() {
    local repo_path="$1"
    local repo_name="$2"
    local output_path="$3"
    
    if is_repo_processed "$output_path"; then
        echo "Skipping repository: $repo_name"
        return 0
    fi
    
    echo "Processing repository: $repo_name"
    echo "  Source: $repo_path"
    echo "  Output: $output_path"
    
    mkdir -p "$output_path"
    
    cd "$SCRIPT_DIR"
    python ast_parser.py --repo_path "$repo_path" \
        --output_dir "$output_path" || {
        echo "  Error: Failed to process $repo_name"
        return 1
    }
    
    echo "   Completed: $repo_name"
    echo ""
}

if [[ "$DATASET" == "RepoExec" ]]; then
    echo "Processing RepoExec repositories..."
    echo "Looking for repositories in: $REPOS_DIR"
    for repo_dir in "$REPOS_DIR"/*; do
        if [[ -d "$repo_dir" ]]; then
            repo_name="$(basename "$repo_dir")"
            output_path="$OUTPUT_DIR/$repo_name"
            
            run_ast_parser "$repo_dir" "$repo_name" "$output_path"
        fi
    done
    
elif [[ "$DATASET" == "DevEval" ]]; then
    echo "Processing DevEval repositories..."
    echo "Looking for topics in: $REPOS_DIR"
    
    for topic_dir in "$REPOS_DIR"/*; do
        if [[ -d "$topic_dir" ]]; then
            topic_name="$(basename "$topic_dir")"
            echo "Processing topic: $topic_name"
            
            for repo_dir in "$topic_dir"/*; do
                if [[ -d "$repo_dir" ]]; then
                    repo_name="$(basename "$repo_dir")"
                    output_path="$OUTPUT_DIR/$repo_name"
                    
                    run_ast_parser "$repo_dir" "$repo_name" "$output_path"
                fi
            done
        fi
    done

elif [[ "$DATASET" == "versionexec_old" ]]; then
    echo "Processing versionexec_old libraries..."
    echo "Looking for libraries in: $REPOS_DIR"
    
    for repo_dir in "$REPOS_DIR"/*; do
        if [[ -d "$repo_dir" ]]; then
            dir_name="$(basename "$repo_dir")"
            
            if [[ "$dir_name" == *"dist-info"* ]]; then
                echo "Skipping dist-info directory: $dir_name"
                continue
            fi
            
            repo_name="$(normalize_package_name "$dir_name")"
            output_path="$OUTPUT_DIR/$repo_name"
            
            run_ast_parser "$repo_dir" "$repo_name" "$output_path"
        fi
    done

elif [[ "$DATASET" == "versionexec_new" ]]; then
    echo "Processing versionexec_new libraries..."
    echo "Looking for libraries in: $REPOS_DIR"
    
    for repo_dir in "$REPOS_DIR"/*; do
        if [[ -d "$repo_dir" ]]; then
            dir_name="$(basename "$repo_dir")"
            
            if [[ "$dir_name" == *"dist-info"* ]]; then
                echo "Skipping dist-info directory: $dir_name"
                continue
            fi
            
            repo_name="$(normalize_package_name "$dir_name")"
            output_path="$OUTPUT_DIR/$repo_name"
            
            run_ast_parser "$repo_dir" "$repo_name" "$output_path"
        fi
    done
fi

echo "All repositories processed successfully!"
echo "Output saved to: $OUTPUT_DIR"
