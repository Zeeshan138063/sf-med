#!/bin/bash

# ───────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────
IMAGE_NAME="public.ecr.aws/d8i8o7p9/snorefox/snorefox-med"
TAG="latest"  # Change this if you want to version it
PLATFORM="linux/amd64"


# ───────────────────────────────────────────────
# CHECK FOR BUILDX SUPPORT
# ───────────────────────────────────────────────
if ! docker buildx version > /dev/null 2>&1; then
  echo "docker buildx is not installed or enabled."
  echo "Run: docker buildx create --use"
  exit 1
fi

# ───────────────────────────────────────────────
# AUTHENTICATING
# ───────────────────────────────────────────────
echo "Authenticating with Amazon ECR Public..."
aws ecr-public get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin public.ecr.aws || {
  echo "Failed to authenticate with public ECR"
  exit 1
}

# ───────────────────────────────────────────────
# BUILD AND PUSH IMAGE
# ───────────────────────────────────────────────
echo "Building image: $IMAGE_NAME:$TAG for $PLATFORM..."

docker buildx build \
  --platform $PLATFORM \
  -t $IMAGE_NAME:$TAG \
  --push .


# ───────────────────────────────────────────────
# DONE
# ───────────────────────────────────────────────
if [ $? -eq 0 ]; then
  echo "Image successfully built and pushed: $IMAGE_NAME:$TAG"
else
  echo "Failed to build or push image."
  exit 1
fi
