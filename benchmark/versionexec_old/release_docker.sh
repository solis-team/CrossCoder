#!/bin/bash

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"
EVALUATE_IMAGE_NAME="${EVALUATE_IMAGE_NAME:-bigcodebench-evaluate}"
GRADIO_IMAGE_NAME="${GRADIO_IMAGE_NAME:-bigcodebench-gradio}"

while getopts "v:" opt; do
  case $opt in
    v)
      version=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

if [ -z "$version" ]; then
  echo "version is required"
  exit 1
fi

export PYTHONPATH=$PWD pytest tests

docker buildx create --name multiplatform-builder --use || true
docker buildx use multiplatform-builder

# Build and push evaluate image
docker buildx build --platform linux/amd64 \
    -f Docker/Evaluate.Dockerfile . \
    -t "${DOCKER_USERNAME}/${EVALUATE_IMAGE_NAME}:$version" \
    -t "${DOCKER_USERNAME}/${EVALUATE_IMAGE_NAME}:latest" \
    --push

# Build and push gradio image
docker buildx build --platform linux/amd64 \
    -f Docker/Gradio.Dockerfile . \
    -t "${DOCKER_USERNAME}/${GRADIO_IMAGE_NAME}:$version" \
    -t "${DOCKER_USERNAME}/${GRADIO_IMAGE_NAME}:latest" \
    --push
