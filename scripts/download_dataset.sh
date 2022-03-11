#!/bin/bash
gdown --id 1UxxVy0kBpEr9-El6SCeCmo_FZuzvWr8v -O data/original/images.tar.gz
tar -xzvf data/original/images.tar.gz -C data/original/
rm data/original/images.tar.gz