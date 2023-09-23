#!/bin/bash -l

case "$1" in
    --py38) docker build - < DockerfileUbuntuPY38 -t metisfl/ubuntu_focal_x86_64_py38
    ;;
    --py39) docker build - < DockerfileUbuntuPY39 -t metisfl/ubuntu_focal_x86_64_py39
    ;;
    --py310) docker build - < DockerfileUbuntuPY310 -t metisfl/ubuntu_focal_x86_64_py310
    ;;
esac
