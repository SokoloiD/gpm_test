#!/bin/bash
./scripts/download_dataset.sh
./scripts/train.sh
./scripts/build_docker_image.sh
./scripts/start_docker.sh
