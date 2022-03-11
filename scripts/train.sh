#!/bin/bash

python3 src/train_clf.py \
  -s model_weights/team_clf.pth \
  -e 20 \
  -d data/original/images_labelling.csv \
  --images_dir data/original/images \
  -t data/preprocessed/team_membership.json