#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_FILE="${SCRIPT_DIR}/versionexec_old.jsonl"

CHECK_ONLY=false
FOLDER_PATH=""
CLEAN_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check_only)
            CHECK_ONLY=true
            shift
            ;;
        --folder)
            if [ -z "$2" ]; then
                echo "Error: --folder requires a path"
                exit 1
            fi
            FOLDER_PATH="$2"
            shift 2
            ;;
        --clean-docker)
            CLEAN_DOCKER=true
            shift
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Usage: $0 [--check_only] [--clean-docker] --folder <path_to_folder>"
            exit 1
            ;;
    esac
done

if [ -z "$FOLDER_PATH" ]; then
    echo "Error: --folder is required"
    echo "Usage: $0 [--check_only] --folder <path_to_folder>"
    exit 1
fi

if [[ "$FOLDER_PATH" != /* ]]; then
    FOLDER_PATH="$(cd "$(dirname "$FOLDER_PATH")" && pwd)/$(basename "$FOLDER_PATH")"
fi
cd "$SCRIPT_DIR"

FOLDER_NAME=$(basename "$FOLDER_PATH")
SANITIZED_FILE="bcb_results/${FOLDER_NAME}_sanitized_calibrated.jsonl"

if [ "$CHECK_ONLY" = true ]; then
    if [ ! -f "$SANITIZED_FILE" ]; then
        echo "Error: Sanitized file not found: $SANITIZED_FILE"
        exit 1
    fi
    echo "Using pre-converted file: $SANITIZED_FILE"
else
    OLD_FILE="${FOLDER_PATH}/versionexec_old.final.generated.jsonl"
    NEW_FILE="${FOLDER_PATH}/versionexec_new.final.generated.jsonl"
    
    if [ -f "$OLD_FILE" ]; then
        GENERATED_FILE="$OLD_FILE"
        echo "Found old format file: $GENERATED_FILE"
    elif [ -f "$NEW_FILE" ]; then
        GENERATED_FILE="$NEW_FILE"
        echo "Found new format file: $GENERATED_FILE"
    else
        echo "Error: Generated file not found. Checked:"
        echo "  - $OLD_FILE"
        echo "  - $NEW_FILE"
        exit 1
    fi

    echo "Converting generated file..."
    python "${SCRIPT_DIR}/normalize_id.py" --generated_file "$GENERATED_FILE"

    if [ ! -f "$SANITIZED_FILE" ]; then
        echo "Error: Sanitized file not found: $SANITIZED_FILE"
        exit 1
    fi

    echo "Found sanitized file: $SANITIZED_FILE"
fi

echo "Extracting task IDs from benchmark file..."
TASK_IDS=$(python -c "
import json
import sys
task_ids = []
with open('$BENCHMARK_FILE', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sample = json.loads(line)
            task_id = sample.get('task_id', '')
            if task_id.startswith('versionexec/'):
                task_num = task_id.replace('versionexec/', '')
                if task_num.isdigit():
                    task_ids.append(task_num)
print(','.join(sorted(set(task_ids), key=int)))
")

if [ -z "$TASK_IDS" ]; then
    echo "Error: No task IDs found in benchmark file"
    exit 1
fi

echo "Task IDs: $TASK_IDS"
SELECTED_IDS=$(echo "$TASK_IDS" | sed 's/\([0-9]*\)/versionexec\/\1/g')
echo "Selected IDs: $SELECTED_IDS"

DOCKER_IMAGE="${DOCKER_IMAGE:-littlebot/bigcodebench-evaluate:latest}"

echo "==============================================="
echo "Mode: RUN existing image from Docker Hub"
echo "Image: ${DOCKER_IMAGE}"
echo "==============================================="

echo ""
echo "Starting evaluation with image from Docker Hub..."

echo "Pulling image with platform linux/amd64..."
docker pull --platform=linux/amd64 "${DOCKER_IMAGE}"

SAMPLES_FILE="$SANITIZED_FILE"

EVAL_RESULTS_CACHE="${SAMPLES_FILE%.jsonl}_eval_results.json"
if [ -f "$EVAL_RESULTS_CACHE" ]; then
    echo "Warning: Found existing eval results cache: $EVAL_RESULTS_CACHE"
    echo "Removing it to avoid conflicts..."
    rm -f "$EVAL_RESULTS_CACHE"
fi

if [ "$CLEAN_DOCKER" = true ]; then
    echo "Cleaning Docker state (networks, build cache)..."
    docker network prune -f >/dev/null 2>&1 || true
    docker builder prune -f >/dev/null 2>&1 || true
    sleep 2
fi

echo ""
echo "Running evaluation..."
docker run --rm --platform=linux/amd64 \
    -v "${SCRIPT_DIR}:/app/data" \
    -v "${SCRIPT_DIR}:/app" \
    -e BIGCODEBENCH_OVERRIDE_PATH=/app/data/versionexec_old.jsonl \
    "${DOCKER_IMAGE}" \
    --split complete \
    --subset full \
    --samples /app/$SAMPLES_FILE \
    --execution local \
    --parallel 10 \
    --selective_evaluate "$SELECTED_IDS" \
    --pass_k 1,3 \
    --calibrated True \
    --no_gt

echo ""
echo "Checking evaluation results..."

EVAL_RESULTS="${SAMPLES_FILE%.jsonl}_eval_results.json"
PASS_K_RESULTS="${SAMPLES_FILE%.jsonl}_pass_at_k.json"

if [ -f "$EVAL_RESULTS" ]; then
    echo "✓ Evaluation results: $EVAL_RESULTS"
fi

if [ -f "$PASS_K_RESULTS" ]; then
    echo "✓ Pass@k results: $PASS_K_RESULTS"
    cat "$PASS_K_RESULTS"
fi

echo ""
echo "==============================================="
echo "Done!"
echo "==============================================="
