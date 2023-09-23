#!/bin/bash -l

case "$1" in
    --py38) IMAGE_NAME="metisfl/ubuntu_focal_x86_64_py38"
    ;;
    --py39) IMAGE_NAME="metisfl/ubuntu_focal_x86_64_py39"
    ;;
    --py310) IMAGE_NAME="metisfl/ubuntu_focal_x86_64_py310"
    ;;
esac

echo "Using MetisFL Image:" $IMAGE_NAME

# Run & Build using the directory above as container's volume.
docker run -dit -v ./..:/metisfl --name metisfl $IMAGE_NAME
docker exec metisfl python /metisfl/setup.py